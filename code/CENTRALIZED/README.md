# Centralized experiments

The `centralized.py` script contains the code to reproduce [ROCKET](https://link.springer.com/article/10.1007/s10618-020-00701-z) experiments and get the same results.

To run the experiments: `python3 centralized.py`. 
This script will run all the experiments with 5 different seeds for reproducibility, and on all the 128 UCR TCS dataset.

If you want to track metrics with WandB, run `python3 centralized.py --use-wandb`, after modifying entity and project according to your WandB credentials.
