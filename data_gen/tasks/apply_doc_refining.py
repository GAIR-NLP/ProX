import argparse
import os
import random

from datasets import Dataset
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data_utils import get_adapter_func, normalize_lined_text, split_into_batches
from utils.doc_utils import execute_meta_operations
from vllm import LLM, SamplingParams

from data_gen.configs import GentaskConfig

random.seed(42)


# dummy env constants for multi-gpu & multi-node
NODE_GPUS = int(os.environ.get("NODE_GPUS", 8))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))
CUDA_DEVICE = int(os.environ["CUDA_VISIBLE_DEVICES"])
TOTAL_SPLIT = int(os.environ["TOTAL_SPLIT"])


def main(args):
    # load config
    config = GentaskConfig().from_yaml(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.token_template, use_fast=False)
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.bos_token = base_tokenizer.bos_token
    tokenizer.eos_token = base_tokenizer.eos_token

    # prepare data
    if args.data_format == "parquet":
        data_reader = ParquetReader(
            data_folder=config.data_path,
            file_progress=True,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    elif args.data_format == "jsonl.gz":
        data_reader = JsonlReader(
            data_folder=config.data_path,
            file_progress=True,
            adapter=get_adapter_func(args.dataset_name),
            limit=args.limit,
        )

    arguments = []
    for idx, doc in enumerate(
        data_reader.run(
            rank=CUDA_DEVICE + NODE_RANK * NODE_GPUS, world_size=TOTAL_SPLIT
        )
    ):
        arguments.append({"text": doc.text})

    dir_path = os.path.join(
        config.save_path, config.save_name + f"_{CUDA_DEVICE + NODE_RANK * NODE_GPUS}"
    )
    os.makedirs(dir_path, exist_ok=True)
    base_name = config.save_name
    batches = split_into_batches(arguments, config.save_interval)

    tp_size = 1
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=2000)
    engine = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
    )

    for i, batch in enumerate(tqdm(batches)):
        rets = []
        for idx, sample in enumerate(tqdm(batch, total=len(batch), unit="tokenizing")):
            user_msg = sample["text"]
            total_msg = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful, respectful and honest assistant.",
                    },
                    {"role": "user", "content": user_msg},
                ],
                add_generation_prompt=True,
                # tokenize=False,
                truncation=True,
                max_length=2000,
            )
            rets.append(total_msg)
        outputs = engine.generate(
            sampling_params=sampling_params, prompt_token_ids=rets
        )
        outputs = [item.outputs[0].text.strip(" ") for item in outputs]
        rets = [
            {
                "raw_content": sample["text"],
                "text": execute_meta_operations(sample["text"], output),
                "metadata": {
                    "doc_program": output,
                },
            }
            for sample, output in zip(batch, outputs)
        ]

        intermediate_ds = Dataset.from_list(rets)

        out_path = (
            os.path.join(
                dir_path,
                f"{base_name}_{i + 1}_{(len(arguments) - 1) // config.save_interval + 1}.parquet",
            )
            if config.save_interval is not None
            else os.path.join(dir_path, f"{base_name}.parquet")
        )
        intermediate_ds.to_parquet(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--token_template",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument("--batch_size", type=int, default=1000000)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of samples to process, for debugging.",
    )
    parser.add_argument("--data_format", type=str, default="parquet")
    parser.add_argument("--dataset_name", type=str, default="redpajama-v2")
    args = parser.parse_args()
    main(args)
