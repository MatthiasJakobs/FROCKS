# Federated ROCKET

This folder contains the code for running the federated ROCKET baseline experiments.

- Install the OpenFL framework: OpenFL
- Run the `./start_federation.sh` inside `FEDERATED ROCKET (BASELINE)/workspace`, after specifying the correct path to director, envoys, workspace, datasets, and the names of your running machines (localhost for a simulated federation).
Note that it may be necessary to increase the sleep seconds before running the script. This can happen when the number of ROCKET kernels is too high, so the Director and Envoy entities need more time to get alive.
  
- For tracking metrics with WandB, replace the OpenFL source files `openfl/component/collaborator/collaborator.py` and `openfl/component/aggregator/aggregator.py` with the provided `collaborator.py` and `aggregator.py`.
