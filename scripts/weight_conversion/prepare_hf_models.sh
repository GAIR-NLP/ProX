#!/bin/bash

# RUNNING HELP:
# bash scripts/weight_conversion/prepare_hf_models.sh [raw_model_dir] [out_model_dir] [model_name] [orig_checkpoint_name] [target_name] [scripts_dir]
raw_model_dir=${1}
out_model_dir=${2}
model_name=${3}
orig_checkpoint_name=${4}
target_name=${5}
scripts_dir=${TINYLM_WORK_DIR}

# weight conversion
PYTHONPATH="${PYTHONPATH}:${scripts_dir}/train" python -m scripts.weight_conversion.convert_lit_checkpoint \
    --checkpoint_name ${orig_checkpoint_name}.pth \
    --out_dir $raw_model_dir \
    --model_name $model_name \
    --model_only False # add this during first conversion

wait
sleep 5

# create the output directory and move the files
mkdir -p ${out_model_dir}/${target_name}
mv ${raw_model_dir}/${orig_checkpoint_name}.bin ${out_model_dir}/${target_name}
mv ${out_model_dir}/${target_name}/${orig_checkpoint_name}.bin ${out_model_dir}/${target_name}/pytorch_model.bin
echo ${out_model_dir}/${target_name}/${orig_checkpoint_name}.bin
mv ${raw_model_dir}/config.json ${out_model_dir}/${target_name}

# download the config file
cd ${out_model_dir}/${target_name}
HF_ENDPOINT=https://huggingface.co
if [ "$model_name" = "tiny_LLaMA_0_3b" ]; then
    # download and overwrite the json file
    cp $scripts_dir/vocab_files/llama_hf/* .  
fi
if [ "$model_name" = "tiny_LLaMA_0_7b" ]; then
    # download and overwrite the json file
    cp $scripts_dir/vocab_files/llama_hf/* .    
fi
if [ "$model_name" = "tiny_LLaMA_1b" ]; then
    cp $scripts_dir/vocab_files/llama_hf/* .  
fi
if [ "$model_name" = "tiny_LLaMA_1_7b" ]; then
    cp $scripts_dir/vocab_files/llama_hf/* .    
fi
if [ "$model_name" = "tiny_LLaMA_3b" ]; then
    cp $scripts_dir/vocab_files/llama_hf/* .    
fi
if [ "$model_name" == "Mistral-7B-v0.1" ]; then
    cp $scripts_dir/vocab_files/mistral_hf/* .
fi
if echo "$model_name" | tr '[:upper:]' '[:lower:]' | grep -q "codellama"; then
    cp $scripts_dir/vocab_files/codellama_hf/* .
fi

cd ../../

# save the pretrained model as safetensors using bitandbytes
cd ${scripts_dir}
python -m scripts.weight_conversion.save_pretrained --model_path ${out_model_dir}/${target_name}
# conda activate tinylm

echo 😊 Weight Conversion Done!
