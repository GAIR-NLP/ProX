#!/bin/bash
#SBATCH --job-name=cpt_llama_2_7b_owm_prox
#SBATCH --output=<expected_output_file>
#SBATCH --partition=<your_partition>
#SBATCH --error=<expected_error_file>
#SBATCH --time=50:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --exclusive


# setup env 
chmod +x setup_personal_env.sh
chmod +x setup_common_env.sh
source setup_personal_env.sh
source setup_common_env.sh

# activate conda env
source $TINYLM_CONDA_DIR
conda activate $TINYLM_ENV_NAME

# enter training dir
cd $TINYLM_WORK_DIR
cd train

export PYTHONPATH=$PYTHONPATH:$TINYLM_WORK_DIR/train

srun python pretrain/tinyllama.py \
    --config_path configs/math/cpt_llama2_owm_prox.yaml