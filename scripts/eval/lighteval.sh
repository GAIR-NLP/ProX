cd eval
accelerate launch --multi_gpu --num_processes=8 --main_process_port=29600 run_evals_accelerate.py \
    --model_args "pretrained=${1}" \
    --tasks $2 \
    --custom_tasks "${3}" \
    --override_batch_size 4 \
    --output_dir="${4}" \
    --max_samples 1000 \
    --dataset_loading_processes 66
