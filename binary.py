import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from utils import load_dataset, preprocess_data, split_function, RocketKernel, transform_seeds, get_binary_dataset_names

# Constants
n_clients = 4 
n_kernels = 1000
n_rounds = 10
n_important_weights = (n_kernels * 1) // n_clients
EXPERIMENT_SEEDS = [1,2,3,4,5]

def main():
    #list_of_datasets = get_binary_dataset_names()
    list_of_datasets = ['Chinatown']

    for ds_name in list_of_datasets:
        print('DATASET', ds_name)
        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

        n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        for experiment_seed in EXPERIMENT_SEEDS:
            rng = np.random.RandomState(experiment_seed)

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
                    print(client_id, f1_score(lm.predict(x_test_transformed), y_test))

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
                print(f'agg f1 ({epoch+1})', f1_score(ag_lin.predict(X_test_transformed), Y_test))

                used_seeds = unique_seeds

                # Check for convergence
                if epoch == 0:
                    old_weights = new_weights.reshape(-1).copy()
                else:
                    nw = new_weights.reshape(-1)
                    if len(nw) == len(old_weights) and np.all(np.isclose(old_weights, nw)):
                        print('Converged after epoch', epoch+1, 'with', len(used_seeds), 'kernels')
                        break
                    old_weights = new_weights.copy()

            # Evaluate on test
            final_prediction = ag_lin.predict(X_test_transformed)
            print('final', f1_score(final_prediction, Y_test))

if __name__ == '__main__':
    main()