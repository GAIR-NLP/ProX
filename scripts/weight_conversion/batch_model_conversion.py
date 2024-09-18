# scripts for automatic weight conversion from litgpt => hf
import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--litgpt_model_dir", type=str, required=True)
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument(
        "--save_token_interval",
        type=int,
        default=1,
        help="the training tokens interval to save the model. Default is 1 (billion) tokens.",
    )
    parser.add_argument(
        "--arch_name",
        type=str,
        default="tiny_LLaMA_0_7b",
        help="the model name to be converted. Default is tiny_LLaMA_0_7b.",
    )
    parser.add_argument(
        "--specific_model_steps",
        type=str,
        default=None,
        help="specific model(s) to be converted. Default is None. e.g., 10000,20000,30000",
    )
    parser.add_argument(
        "--base_token",
        type=int,
        default=0,
        help="the base token to start counting the training tokens. Default is 0 (billion).",
    )
    args = parser.parse_args()

    PT_MODEL_OUTPUT_DIR = os.environ.get("PT_MODEL_OUTPUT_DIR")
    HF_MODEL_OUTPUT_DIR = os.environ.get("HF_MODEL_OUTPUT_DIR")
    litgpt_model_dir = os.path.join(PT_MODEL_OUTPUT_DIR, args.litgpt_model_dir)
    hf_model_dir = os.path.join(HF_MODEL_OUTPUT_DIR, args.hf_model_dir)

    # the model pth will looks like:
    # /nas/shared/GAIR/fan/out/tinyllama_0_7b_cpt_iter2_25B/iter-160000-ckpt.pth
    # sort model according to step number `xxxxxx` extracted from `iter-xxxxxx-ckpt.pth``
    model_info_list = []
    for file in os.listdir(litgpt_model_dir):
        if file.endswith(".pth"):
            step = int(file.split(".")[0].split("-")[1])
            model_info_list.append((step, file))
    model_info_list = sorted(model_info_list, key=lambda x: x[0])

    if args.specific_model_steps is not None:
        specific_model_steps = [
            int(step) for step in args.specific_model_steps.split(",")
        ]

    for idx, model_info in enumerate(model_info_list):
        if (
            args.specific_model_steps is not None
            and model_info[0] not in specific_model_steps
        ):
            print(model_info[0], specific_model_steps)
            continue

        # extract only file prefix
        model_name = model_info[1].split(".")[0]
        training_token = (idx + 1) * args.save_token_interval + args.base_token
        hf_model_name = f"{training_token}B"
        print(
            f"Converting model {model_name}.pth ({training_token}B tokens) to HF format."
        )
        # convert model to HF format bash scripts
        os.system(
            f"bash scripts/weight_conversion/prepare_hf_models.sh {litgpt_model_dir} {hf_model_dir} {args.arch_name} {model_name} {hf_model_name}"
        )
        print(
            f"Model {model_name} converted to HF format {hf_model_dir}/{hf_model_name}."
        )

    print("All models converted to HF format.")
