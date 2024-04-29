#!/bin/bash

echo "Cleaning existing fx instances..."
ssh MACHINE1 "pkill fx ; cd PATH_TO_WORKSPACE ; rm -rf __pycache__ *.pkl save requirements.txt output"
ssh MACHINE2 "pkill fx ; pkill start_director.py; cd PATH_TO_DIRECTOR ; rm -rf __pycache__ .DS_Store output UCR*"
ssh MACHINE3 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
ssh MACHINE4 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
ssh MACHINE5 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
ssh MACHINE6 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"

datasets=$(ls "PATH_TO_UCR_ARCHIVE")

#check wandb runs
wandb_runs=$(python wandb_runs.py)

# extract wandb already done experiments
run_names=$(echo "$wandb_runs" | awk '/RUN_NAMES:/ { for (i = 2; i <= NF; i++) print $i }')

for dataset in $datasets
do
    run_name="Aggregator_$dataset"
    if [ -z "${run_names##*$run_name*}" ]; then
        echo "Experiment $run_name already executed."
    else
        for i in {0..4}
        do
            echo "DATASET: $dataset"
            rocket=1 #0 false, 1 true
            kernels=10000
            echo "Starting up fx instances..."
            ssh MACHINE1 "source /home/ubuntu/anaconda3/bin/activate ; conda activate PATH_TO_VIRTUALENV ; cd PATH_TO_DIRECTOR ; nohup python3 start_director.py $dataset $rocket $kernels $i > output_director 2>&1 &" &
            ssh MACHINE2 "source /home/ubuntu/anaconda3/bin/activate ; conda activate PATH_TO_VIRTUALENV ; cd PATH_TO_ENVOY ; sleep 200 ; nohup python3 start_envoy.py $dataset 1 $rocket $kernels $i > output_envoy1 2>&1 &" &
            ssh MACHINE3 "source /home/ubuntu/anaconda3/bin/activate ; conda activate PATH_TO_VIRTUALENV ; cd PATH_TO_ENVOY ; sleep 200 ; nohup python3 start_envoy.py $dataset 2 $rocket $kernels $i > output_envoy2 2>&1 &" &
            ssh MACHINE4 "source /home/ubuntu/anaconda3/bin/activate ; conda activate PATH_TO_VIRTUALENV ; cd PATH_TO_ENVOY ; sleep 200 ; nohup python3 start_envoy.py $dataset 3 $rocket $kernels $i > output_envoy3 2>&1 &" &
            ssh MACHINE5 "source /home/ubuntu/anaconda3/bin/activate ; conda activate PATH_TO_VIRTUALENV ; cd PATH_TO_ENVOY ; sleep 200 ; nohup python3 start_envoy.py $dataset 4 $rocket $kernels $i > output_envoy4 2>&1 &" &
            
            echo "Starting experiment..."
            sleep 1100
            python federated_ucr_workspace.py $i $dataset
            sleep 1100

            echo "Cleaning existing fx instances..."
            ssh worker-1 "pkill fx ; cd PATH_TO_WORKSPACE ; rm -rf __pycache__ *.pkl save requirements.txt output"
            ssh worker-1 "pkill fx ; cd PATH_TO_DIRECTOR ; rm -rf __pycache__ .DS_Store output"
            ssh worker-2 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
            ssh worker-3 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
            ssh worker-4 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
            ssh worker-5 "pkill fx ; cd PATH_TO_ENVOY ; rm -rf __pycache__ .DS_Store"
            
        done
    fi
done
echo "Done!"