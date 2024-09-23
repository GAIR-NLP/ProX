import gzip
import json
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import train.lit_gpt.packed_dataset as packed_dataset
from train.lit_gpt import Tokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str = "train",
    filenames_subset: List[str] = None,
    process_id: int = 0,
    text_key: str = "text",
) -> None:
    destination_path.mkdir(exist_ok=True, parents=True)
    print(f"dumping tokens to {destination_path}")

    tokenizer = Tokenizer(tokenizer_path)
    auto_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    filenames = filenames_subset
    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_web_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in tqdm(filenames, total=len(filenames)):
        print(f"Processing {filepath}")
        if filepath.endswith(
            ".json.gz"
        ):  # this happans when we are processing the raw redpj data
            try:
                contents = pd.read_json(filepath, lines=True, compression="gzip")[
                    "raw_content"
                ]
            except Exception as e:
                print(e)
                print(f"Error reading {filepath}!!")
                continue
        elif filepath.endswith(".jsonl.gz"):
            try:
                contents = []
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    for line in f:
                        contents.append(json.loads(line)[text_key])
            except Exception as e:
                print(e)
                print(f"Error reading {filepath}!!")
        elif filepath.endswith(".parquet"):
            try:
                contents = pd.read_parquet(filepath, engine="pyarrow")[text_key]
            except Exception as e:
                print(e)
                print(f"Error reading {filepath}!!")
                continue
        else:
            raise NotImplementedError(f"File type not supported: {filepath}")

        # Tokenize and write to builder
        for idx, text in tqdm(enumerate(contents), total=len(contents)):
            if text == "":
                continue

            # do not keep the bos token
            auto_text_ids = auto_tokenizer.encode(text=text) + [tokenizer.eos_id]
            if auto_text_ids[0] == tokenizer.bos_id:
                auto_text_ids = auto_text_ids[1:]

            auto_text_ids = torch.tensor(auto_text_ids, dtype=torch.int)
            builder.add_array(np.array(auto_text_ids, dtype=builder.dtype))
            # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details

        del contents


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str = "train",
    percentage: float = 1.0,
    text_key: str = "text",
    filenames_subset: List[str] = None,
) -> None:
    filenames = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if (
                file.endswith(".parquet")
                or file.endswith(".json.gz")
                or file.endswith(".jsonl.gz")
            ):
                filenames.append(os.path.join(root, file))

    # only retrain subsets that follow the prefix in filenames_subset
    if filenames_subset:
        filenames = [
            f
            for f in filenames
            if any([f.endswith(prefix) for prefix in filenames_subset])
        ]
    filenames = filenames[: int(len(filenames) * percentage)]
    print(f"Processing {len(filenames)} files")
    # HARD CODED: we only use 1/2 of the cpu cores to avoid OOM on our clusters @fan
    num_processes = min(len(filenames), os.cpu_count() // 2)
    print(len(filenames), num_processes)
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(
            target=prepare_full,
            args=(
                source_path,
                tokenizer_path,
                destination_path,
                chunk_size,
                split,
                list(subset),
                i,
                text_key,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
