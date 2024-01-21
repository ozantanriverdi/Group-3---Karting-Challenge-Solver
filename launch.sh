#!/bin/bash

#SBATCH --job-name=ppo_chunk_nr_1
#SBATCH --partition=All
#SBATCH --nodes=1
#SBATCH --mincpus=5
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/logs/out_job_%j_name_%x.log
#SBATCH --error=./slurm/logs/err_job_%j_name_%x.log

# REPLACE WITH YOUR NAME
mkdir /tmp/engeln
export TMP="/tmp/engeln"
export TMPDIR="/tmp/engeln"

# Switch to the 'cip-pool-training' branch
git checkout cip-pool-training

# Fetch the latest changes
git fetch origin cip-pool-training

# Check if there are any changes to pull
if git diff --quiet HEAD FETCH_HEAD; then
    echo "No new changes to pull"
else
    # Pull the latest changes and handle errors
    if ! git pull origin cip-pool-training; then
        echo "Error: Failed to pull changes"
        exit 1
    fi
fi

# Check for merge conflicts
if git diff --name-only --diff-filter=U | grep -q .; then
    echo "Error: Merge conflicts exist. Please resolve them."
    exit 1
fi


# Default values
default_learning_rate=0.001
default_discount_factor=0.99
default_epsilon=0.1
default_gae_lambda=0.95
default_value_function_coef=0.5
default_num_episodes=300
default_num_epochs=10
default_batch_size=64
default_horizon=512
default_entropy_coefficient_init=0.1
default_entropy_coefficient_target=0.01
default_entropy_coefficient_const=0.01
default_entropy_decay_steps=100000

default_agent="PPO"
default_environment="ORIGINAL"
default_entropy_setting="FIXED"
default_plotting_setting="REWARDS_AND_TRACK_DISTANCE"
default_zero_brake=False
default_automatic_restart=False
default_use_existing_model=""
default_seed=42


# Set the values passed as arguments
learning_rate=$default_learning_rate
discount_factor=$default_discount_factor
epsilon=$default_epsilon
gae_lambda=$default_gae_lambda
value_function_coef=$default_value_function_coef
num_episodes=$default_num_episodes
num_epochs=$default_num_epochs
batch_size=$default_batch_size
horizon=$default_horizon
entropy_coefficient_init=$default_entropy_coefficient_init
entropy_coefficient_target=$default_entropy_coefficient_target
entropy_coefficient_const=$default_entropy_coefficient_const
entropy_decay_steps=$default_entropy_decay_steps

agent=$default_agent
environment=$default_environment
entropy_setting=$default_entropy_setting
plotting_setting=$default_plotting_setting
zero_brake=$default_zero_brake
automatic_restart=$default_automatic_restart
use_existing_model=$default_use_existing_model
seed=$default_seed


# Parse the arguments (if provided)
for arg in "$@"; do
  case $arg in
    # hyperparameters
    learning_rate=*)
      learning_rate="${arg#*=}"
      ;;
    discount_factor=*)
      discount_factor="${arg#*=}"
      ;;
    epsilon=*)
      epsilon="${arg#*=}"
      ;;
    gae_lambda=*)
      gae_lambda="${arg#*=}"
      ;;
    value_function_coef=*)
      value_function_coef="${arg#*=}"
      ;;
    num_episodes=*)
      num_episodes="${arg#*=}"
      ;;
    num_epochs=*)
      num_epochs="${arg#*=}"
      ;;
    batch_size=*)
      batch_size="${arg#*=}"
      ;;
    horizon=*)
      horizon="${arg#*=}"
      ;;
    entropy_coefficient_init=*)
      entropy_coefficient_init="${arg#*=}"
      ;;
    entropy_coefficient_target=*)
      entropy_coefficient_target="${arg#*=}"
      ;;
    entropy_coefficient_const=*)
      entropy_coefficient_const="${arg#*=}"
      ;;
    entropy_decay_steps=*)
      entropy_decay_steps="${arg#*=}"
      ;;
    # run config
    agent=*)
      agent="${arg#*=}"
      ;;
    environment=*)
      environment="${arg#*=}"
      ;;
    entropy_setting=*)
      entropy_setting="${arg#*=}"
      ;;
    plotting_setting=*)
      plotting_setting="${arg#*=}"
      ;;
    zero_brake=*)
      zero_brake="${arg#*=}"
      ;;
    automatic_restart=*)
      automatic_restart="${arg#*=}"
      ;;
    use_existing_model=*)
      use_existing_model="${arg#*=}"
      ;;   
    seed=*)
      seed="${arg#*=}"
      ;;
  esac
done

# Update the hyperparameters.py file with the new values
cat <<EOF >configs/hyperparameters.py
hyperparameters = {
  "learning_rate": $learning_rate,
  "discount_factor": $discount_factor,
  "epsilon": $epsilon,
  "gae_lambda": $gae_lambda,
  "value_function_coef": $value_function_coef,
  "num_episodes": $num_episodes,
  "num_epochs": $num_epochs,
  "batch_size": $batch_size,
  "horizon": $horizon,
  "entropy_coefficient_init": $entropy_coefficient_init,
  "entropy_coefficient_target": $entropy_coefficient_target,
  "entropy_coefficient_const": $entropy_coefficient_const,
  "entropy_decay_steps": $entropy_decay_steps
}
EOF

# Update the run_config.py file with the new values
cat <<EOF >configs/run_config.py
config = {
    "agent": "$agent",
    "environment": "$environment",
    "entropy_setting": "$entropy_setting",
    "plotting_setting": "$plotting_setting",
    "zero_brake": $zero_brake,
    "automatic_restart": $automatic_restart,
    "use_existing_model": "$use_existing_model",
    "seed": $seed
}
EOF


source ./myenv/bin/activate

python3 main.py --no-graphics
# Reset the hyperparameters to default values
cat <<EOF >configs/hyperparameters.py
hyperparameters = {
  "learning_rate": $default_learning_rate,
  "discount_factor": $default_discount_factor,
  "epsilon": $default_epsilon,
  "gae_lambda": $default_gae_lambda,
  "value_function_coef": $default_value_function_coef,
  "num_episodes": $default_num_episodes,
  "num_epochs": $default_num_epochs,
  "batch_size": $default_batch_size,
  "horizon": $default_horizon,
  "entropy_coefficient_init": $default_entropy_coefficient_init,
  "entropy_coefficient_target": $default_entropy_coefficient_target,
  "entropy_coefficient_const": $default_entropy_coefficient_const,
  "entropy_decay_steps": $default_entropy_decay_steps
}
EOF

# Reset the run_config to default values
cat <<EOF >configs/run_config.py
from utils.enums import AgentEnum, EnvironmentEnum, EntropySettingEnum, PlottingSettingEnum

config = {
    "agent": AgentEnum.PPO, # PPO | PPOBETA | SB_PPO | RANDOM
    "environment": EnvironmentEnum.ORIGINAL, # ORIGINAL | ADVANCED
    "entropy_setting": EntropySettingEnum.FIXED, # FIXED | SCHEDULED
    "plotting_setting": PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE, # REWARDS | TRACK_DISTANCE | REWARDS_AND_TRACK_DISTANCE
    "zero_brake": False,  # False | True
    "automatic_restart": False, # False | True
    "use_existing_model": "", # name of torch model (e.g: 20230626_162416_model)
    "seed": $default_seed # False | True
}
EOF


# # commit results
# git add logger/logs models
# git commit -m "training results from cip pool"
git stash

# Fetch the latest changes
git fetch origin cip-pool-training

# Check if there are any changes to pull
if git diff --quiet HEAD FETCH_HEAD; then
    echo "No new changes to pull"
else
    # Pull the latest changes and handle errors
    if ! git pull origin cip-pool-training; then
        echo "Error: Failed to pull changes"
        exit 1
    fi
fi

git stash pop
git add logger/logs models
git commit -m "training results from cip pool"
git push

