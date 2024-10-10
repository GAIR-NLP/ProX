import hashlib
from functools import partial

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")


def generate_unique_id(string):
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def redpajama_adapter(data: dict, source_file, id_in_file, token_limit):
    if (
        token_limit and len(tokenizer.encode(data["raw_content"])) <= token_limit
    ) or token_limit is None:
        data["text"] = data["raw_content"]
    else:
        data["text"] = ""
    if "itermediate_results" in data:
        del data["itermediate_results"]

    ret = {
        "text": data["text"],
        "id": generate_unique_id(data["text"]),
        "media": data.pop("media", []),
        "metadata": data.pop("meta", {}) | data,
    }
    return ret


def openwebmath_adapter(data: dict, source_file, id_in_file, line_num_limit):
    if (
        line_num_limit and len(data["text"].split("\n")) <= line_num_limit
    ) or line_num_limit is None:
        data["text"] = data["text"]
    else:
        data["text"] = ""

    del data["metadata"]

    return {
        "text": data["text"],
        "id": 0,
        "media": data.pop("media", []),
        "metadata": data.pop("metadata", {}) | data,
    }


def get_adapter_func(dataset: str, token_limit=None, line_num_limit=None):
    func_map = {
        "redpajama-v2": partial(redpajama_adapter, token_limit=token_limit),
        "openwebmath": partial(openwebmath_adapter, line_num_limit=line_num_limit),
        "c4": None,
    }
    return func_map[dataset]


def split_into_batches(arguments, batch_size):
    """Split arguments into batches according to batch_size."""
    if batch_size is None or batch_size <= 0:
        return [arguments]
    else:
        return [
            arguments[i : i + batch_size] for i in range(0, len(arguments), batch_size)
        ]


def normalize_lined_text(text: str, max_token=1000) -> str:
    text_line = []
    max_digits = 3
    cur_tokens = 0
    for idx_line, line in enumerate(text.split("\n")):
        normalize_line = f"[{idx_line:0{max_digits}d}]{line}"
        line_token = len(tokenizer.encode(normalize_line))
        if line_token + cur_tokens > max_token:
            trunc_line = tokenizer.decode(
                tokenizer.encode(
                    line, max_length=max_token - cur_tokens, truncation=True
                )
            ).strip(tokenizer.bos_token + " ")
            text_line.append(f"[{idx_line:0{max_digits}d}]{trunc_line}")
            break
        else:
            text_line.append(f"[{idx_line:0{max_digits}d}]{line}")
            cur_tokens += line_token
    # we will thus skip some documents contains only one line > max_token
    if len(text_line) == 0:
        text = tokenizer.decode(
            tokenizer.encode(text, max_length=max_token, truncation=True)
        ).strip(tokenizer.bos_token + " ")
    else:
        text = "\n".join(text_line)
    return text
