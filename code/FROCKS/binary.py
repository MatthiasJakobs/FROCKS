import pandas as pd
import numpy as np
import wandb
import argparse
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from utils import load_dataset, preprocess_data, split_function, RocketKernel, transform_seeds, get_binary_dataset_names

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clients', type=int, default=4, help='Choose the number of parties of the federation')
parser.add_argument('-k', '--kernels', type=int, default=1000, help='Choose the number of ROCKET kernels to use')
parser.add_argument('-r', '--rounds', type=int, default=100, help='Choose the number of rounds of federated training')
parser.add_argument('--debug', action='store_true', help='Use this option to disable WandB metrics tracking')
args = parser.parse_args()

# Constants
n_clients = args.clients 
n_kernels = args.kernels
n_rounds = args.rounds
n_important_weights = (n_kernels * 1) // n_clients
EXPERIMENT_SEEDS = [1,2,3,4,5]

def main():
    list_of_datasets = get_binary_dataset_names()
    #list_of_datasets = ['Chinatown']

    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init()

    # All the runs
    run_names = []
    runs = wandb.Api().runs("mlgroup/FROCKS")
    for run in runs:
        run_names.append(run.name)
    print("RUN NAMES DONE: ", run_names)
    wandb.finish()

    for ds_name in list_of_datasets:
        print('DATASET', ds_name)

        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

        n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        for experiment_seed in EXPERIMENT_SEEDS:
            print('SEED', experiment_seed)
            rng = np.random.RandomState(experiment_seed)
            np.random.seed(experiment_seed)
            random.seed(experiment_seed)

            if args.debug:
                wandb.init(mode="disabled")
            else:
                entity = "mlgroup"
                project = "FROCKS"
                run_name = f"{ds_name}_{n_kernels}_KERNELS_{experiment_seed}"
                tags = ["TCS", "FROCKS"]
                if run_name in run_names:
                    print(f"Experiment {run_name} already executed.")
                    continue
                else:
                    wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

            s_X_train, s_Y_train = split_function(X_train, Y_train, n_clients, rng)
            s_X_test, s_Y_test = split_function(X_test, Y_test, n_clients, rng)

            # Initialize different seeds for all clients
            seeds = np.arange(n_clients*n_kernels).reshape(n_clients, n_kernels)

            ts_length = len(X_train[0])
            kernels = [[RocketKernel(seed=int(seed), ts_length=ts_length, ppv_only=True) for seed in seeds[client_id]] for client_id in range(n_clients)]

            intercept = 0
            
            for epoch in range(n_rounds):

                weights = []
                new_used_seeds = []

                for client_id in range(n_clients):
                    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                    x_test, y_test = s_X_test[client_id], s_Y_test[client_id]
                    if epoch == 0:
                        K = kernels[client_id]
                    else:
                        K = [RocketKernel(seed, ts_length) for seed in used_seeds]

                    # Transform data
                    x_train_transformed = np.concatenate([k.transform(x_train) for k in K], axis=1)
                    x_test_transformed = np.concatenate([k.transform(x_test) for k in K], axis=1)

                    lm = LogisticRegression(random_state=rng)
                    lm.fit(x_train_transformed, y_train)
                    client_accuracy = accuracy_score(lm.predict(x_test_transformed), y_test)
                    client_f1 = f1_score(lm.predict(x_test_transformed), y_test)
                    print("Client: ", client_id, 
                        "accuracy: ", client_accuracy,
                        "f1-score: ", client_f1)
                    wandb.run.summary["step"] = epoch
                    wandb.log({f'Client {client_id} Test accuracy with {n_kernels} kernels': client_accuracy,
                               f'Client {client_id} Test F1 with {n_kernels} kernels': client_f1})

                    # Find best weights
                    w = lm.coef_[0]
                    highest_weight_indices = np.argsort(-np.abs(w))[:n_important_weights]
                    intercept += lm.intercept_[0]
                    weights.append(w[highest_weight_indices])
                    if epoch == 0:
                        used_seeds = seeds[client_id]
                    new_used_seeds.append([used_seeds[idx] for idx in highest_weight_indices])

                weights = np.concatenate(weights)
                new_used_seeds = np.concatenate(new_used_seeds)
                intercept = np.array(intercept) / n_clients

                # Average duplicate entries
                unique_seeds = np.unique(new_used_seeds)
                new_weights = np.zeros((len(unique_seeds)))
                for idx, seed_id in enumerate(unique_seeds):
                    new_weights[idx] = weights[np.where(new_used_seeds == seed_id)].mean()

                # Eval
                ag_lin = LogisticRegression()
                ag_lin.is_fitted = True
                ag_lin.coef_ = new_weights.reshape(1, -1)
                ag_lin.intercept_ = intercept
                ag_lin.classes_ = np.arange(n_classes)
                X_test_transformed = transform_seeds(X_test, unique_seeds, ts_length)
                aggregator_accuracy = accuracy_score(ag_lin.predict(X_test_transformed), Y_test)
                aggregator_f1 = f1_score(ag_lin.predict(X_test_transformed), Y_test)
                print(f'Epoch: ({epoch+1})', "Aggregator ",
                    "accuracy: ", aggregator_accuracy,
                    "f1-score: ", aggregator_f1)
                wandb.run.summary["step"] = epoch
                wandb.log({f'Aggregator Test accuracy with {n_kernels} kernels': aggregator_accuracy,
                               f'Aggregator Test F1 with {n_kernels} kernels': aggregator_f1})

                used_seeds = unique_seeds

                # Check for convergence
                if epoch == 0:
                    old_weights = new_weights.reshape(-1).copy()
                else:
                    nw = new_weights.reshape(-1)
                    if len(nw) == len(old_weights) and np.all(np.isclose(old_weights, nw)):
                        print('Converged after epoch', epoch+1, 'with', len(used_seeds), 'kernels')
                        wandb.log({f'Rounds for convergence': epoch+1,
                        f'Convergence kernels': len(used_seeds)})
                        break
                    old_weights = new_weights.copy()

            # Evaluate on test
            final_prediction = ag_lin.predict(X_test_transformed)
            final_accuracy = accuracy_score(final_prediction, Y_test)
            final_f1 = f1_score(final_prediction, Y_test)
            print('Final scores. Accuracy: ', final_accuracy,
                "f1-score: ", final_f1)
            wandb.log({f'Final accuracy': final_accuracy,
                        f'Final F1': final_f1})
            wandb.finish()

if __name__ == '__main__':
    main()