# FROCKS experiments

The `binary.py` script contains the code to reproduce our FROCKS experiments and get the same results.

To run the experiments: `python3 binary.py`. 
This script will run all the experiments with 5 different seeds for reproducibility, and on all the binary classification datasets of the UCR TCS archive.

To disable tracking metrics with WandB, run `python3 binary.py --debug`. By default, WandB tracking is enabled. Adjust the entries of entity and project according to your credentials WandB credentials.

`python3 binary.py --help` for discovering other possible training options.