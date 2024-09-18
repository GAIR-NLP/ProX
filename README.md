# Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale

<p align="center">
  <img src="./static/images/prox-logo.png">
</p>
<a href="https://huggingface.co/gair-prox" target="_blank">
    <img alt="Models" src="https://img.shields.io/badge/🤗-HuggingFace Repo-blue" />
</a>
<a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/📑-Paper-blue" />
</a>
<a href="https://gair-nlp.github.io/program-every-example/" target="_blank">
<img alt="Project Page" src="https://img.shields.io/badge/🧪-Project Page-blue" />
</a>
<a href="https://opensource.org/license/apache-2-0" target="_blank">
    <img alt="License: apache-2-0" src="https://img.shields.io/github/license/saltstack/salt" />
</a>
<a href="https://github.com/GAIR-NLP/program-every-example" target="_blank">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/GAIR-NLP/program-every-example?style=social" />
</a>
<a href="https://github.com/GAIR-NLP/program-every-example/issues" target="_blank">
    <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/GAIR-NLP/program-every-example" />
</a>

## 🔥 News

- **[19 September, 2024]:** 🎉 We open-sourced [pre-training corpus](https://huggingface.co/collections/gair-prox/prox-dataset-66e81c9d560911b836bb3704) curated by our ProX framework, containing > 100B high quality general domain corpus and ~5B high quality math corpus, together with models([ProX](https://huggingface.co/collections/gair-prox/prox-general-models-65f1674f0607712c4d6eec76) and [ProXMath](https://huggingface.co/collections/gair-prox/prox-math-models-66e92c3e5d54b27612286eb9)) trained using these data.

## 🚀 Introduction

## Quick Start

First, we have to install all the libraries listed in requirements.txt

```bash
git clone https://github.com/GAIR-NLP/program-every-example.git prox
cd prox
conda create -n prox python=3.10
conda activate prox
pip install -r requirements.txt
```

For acceleration, we need to install flash-attention with some fused kernels:

<details>
<summary>Click me</summary>
<p>

```bash
pip install flash-attn --no-build-isolation
# this part is quite similar to TinyLlama repo
# you can also refer to its detailed guide at: https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
```

</p>
</details>

Then, we can install lighteval & math-eval for evaluation

<details>
<summary>
<b>lighteval</b>
</summary>
<p>

```bash
git clone https://github.com/huggingface/lighteval.git
cd lighteval
pip install -e .
```

</p>
</details>

<details>
<summary>
<b>math-eval</b>
</summary>
<p>

```bash
#TODO
```

</p>
</details>

## Training on ProX curated data

We provide over 100B high quality general domain corpus and ~5B high quality math corpus. You can train your own model using this data.

Here we provide an example to download, tokenize, train a model using ProX data with litgpt, finally with thorough evaluation.
Feel free to modify the script to fit your own needs.

First step is to setup your environment variables:

```bash
# 1. using setup_personal_env and setup_common_env
source setup_personal_env.sh
source setup_common_env.sh
```

Then you can download the data, and tokenize the data

```bash
# 2. download the data, e.g., RedPajama-pro
python scripts/data_download/hf_download.py \
    --dataset_name gair-prox/RedPajama-pro

# 3. tokenize the data
export PYTHONPATH=$PYTHONPATH:$TINYLM_WORK_DIR/train
python -m train.data_tokenize.prepare_web \
    --source_path $RAW_DATA_DIR/gair-prox/RedPajama-pro \
    --tokenizer_path $TINYLM_WORK_DIR/vocab_files/llama_hf \
    --destination_path $TOKENIZE_DATA_DIR/llama/RedPajama-pro \
    --split train \
    --percentage 1.0
```

You should see many ".bin" files in the destination path. Then you can train a model using the tokenized data.

```bash

```
