#!/bin/bash
#SBATCH --job-name=pt_tlm_s_c4_prox
#SBATCH --output=<expected_output_file>
#SBATCH --partition=<your_partition>
#SBATCH --error=<expected_error_file>
#SBATCH --time=10:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32


# setup env 
chmod +x setup_personal_env.sh
chmod +x setup_common_env.sh
source setup_personal_env.sh
source setup_common_env.sh

# activate conda env
source $TINYLM_CONDA_DIR
conda activate $TINYLM_ENV_NAME

# enter training dir
echo "Current working directory: $(pwd)"
cd $TINYLM_WORK_DIR
cd train

export PYTHONPATH=$PYTHONPATH:$TINYLM_WORK_DIR/train

srun python pretrain/tinyllama.py \
    --config_path configs/general/pt_tlm_m_c4_prox.yaml

cd $TINYLM_WORK_DIR
conda activate tinylm
python -m scripts.weight_conversion.batch_model_conversion \
    --litgpt_model_dir pt_llama_1_7b_c4_50B_prox \
    --hf_model_dir pt_llama_1_7b_c4_50B_prox \
    --save_token_interval 5 \
    --arch_name tiny_LLaMA_1_7b

# model step list is from 1 ~ 25
export NNODES=8
export TOTAL_STEPS=50
export SAVE_STEP_INTERVAL=5

cd $TINYLM_WORK_DIR
source $TINYLM_CONDA_DIR
conda activate lmeval

cmd='
TOTAL_STEPS=$TOTAL_STEPS
node_id=$SLURM_PROCID
total_nodes=$SLURM_NTASKS

echo "Node $node_id of $total_nodes starting"

# Function to get the nth step for this node
get_step_for_node() {
    local n=$1
    echo $(((n * total_nodes + node_id + 1) * SAVE_STEP_INTERVAL))
}

# Generate the list of steps for this node
node_steps=""
n=0
while true; do
    step=$(get_step_for_node $n)

    if (( step > TOTAL_STEPS )); then
        break
    fi
    if [ -z "$node_steps" ]; then
        node_steps="$step"
    else
        node_steps="$node_steps,$step"
    fi
    n=$((n + 1))
done

echo "Steps for node $node_id: $node_steps"
export HUGGINGFACE_HUB_CACHE=$DATA_ROOT_DIR/huggingface/hub
python -m scripts.eval.base_evaluation \
    --hf_model_dir pt_llama_1_7b_c4_50B_prox \
    --task_impl lighteval \
    --task_set fineweb \
    --model_step_list $node_steps
'
echo "Executing command:"
echo "$cmd"

srun --ntasks=$NNODES --ntasks-per-node=1 bash -c "$cmd"
