import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download

# Use argparse to handle command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Dataset name to download",
    default="HuggingFaceFW/fineweb",
    choices=[
        "gair-prox/FineWeb-pro",
        "gair-prox/open-web-math-pro",
        "gair-prox/c4-pro",
        "gair-prox/RedPajama-pro",
        "HuggingFaceFW/fineweb",
        "allenai/c4",
        "EleutherAI/proof-pile-2",
    ],
)
parser.add_argument(
    "--allow_patterns",
    type=str,
    help="Allow patterns to download",
    default=None,
    choices=[
        "sample/350BT/*",  # for downloading fineweb
        "en/*",  # for downloading c4
    ],
)
args = parser.parse_args()
raw_data_dir = os.path.join(os.environ.get("RAW_DATA_DIR"))

snapshot_download(
    repo_id=args.dataset_name,
    allow_patterns=args.allow_patterns,
    repo_type="dataset",
    local_dir=f"{raw_data_dir}/{args.dataset_name}",
    local_dir_use_symlinks=False,
    force_download=True,
)
