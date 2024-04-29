import wandb
wandb.init()
# Elenca i nomi degli esperimenti eseguiti
run_names = []
# Recupera tutti gli esperimenti nel progetto specificato
runs = wandb.Api().runs("mlgroup/Federated ROCKET 10000")
for run in runs:
    run_names.append(run.name)
print("RUN NAMES DONE: ", run_names)
wandb.finish()

# Stampa il valore di run_names in un formato che può essere catturato da uno script bash
print("RUN_NAMES:", run_names)