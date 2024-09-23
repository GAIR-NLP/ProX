import os
import sys
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--task_impl",
        choices=["lighteval", "lmeval"],
        required=True
    )
    parser.add_argument(
        "--task_set",
        choices=["fineweb", "indicator"],
    )
    parser.add_argument(
        "--model_step_list",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--only_gather',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--metric',
        choices=['acc', 'acc_norm'],
        default='acc_norm',
    )
    args = parser.parse_args()

    WORK_DIR = os.environ.get("TINYLM_WORK_DIR")
    HF_MODEL_OUTPUT_DIR = os.environ.get("HF_MODEL_OUTPUT_DIR")
    EVAL_ANALYSIS_DIR = os.environ.get("EVAL_ANALYSIS_DIR")
    hf_model_dir = os.path.join(HF_MODEL_OUTPUT_DIR, args.hf_model_dir)
    eval_analysis_dir = os.path.join(EVAL_ANALYSIS_DIR, args.task_impl, args.task_set, args.hf_model_dir)

    model_info_list = []
    for model_dir in os.listdir(hf_model_dir):
        if model_dir.endswith("B") and os.path.isdir(os.path.join(hf_model_dir, model_dir)):
            step = int(model_dir.split("B")[0])
            model_info_list.append((step, model_dir))
    model_info_list = sorted(model_info_list, key=lambda x: x[0])

    # ===============================
    # do evaluation
    # ===============================
    if not args.only_gather:
        for idx, model_info in enumerate(model_info_list):
            if args.model_step_list is not None:
                model_step_list = args.model_step_list.split(",")
                if str(model_info[0]) not in model_step_list:
                    continue

            full_model_dir = os.path.join(hf_model_dir, model_info[1])
            if args.task_impl == "lighteval":
                print(f"Running lighteval for model {full_model_dir}")
                assert args.task_set in ["fineweb"]
                task_set = f"{WORK_DIR}/eval/{args.task_impl}/{args.task_set}.txt"
                task_impl = f"{WORK_DIR}/eval/{args.task_impl}/tasks.py"
                os.system(f"""bash scripts/eval/lighteval.sh {full_model_dir} {task_set} {task_impl} {eval_analysis_dir}""")
            elif args.task_impl == "lmeval":
                print(f"Running lm eval harness for model {full_model_dir}")
                assert args.task_set in ["indicator"]
                pass
            else:
                raise ValueError("Unknown Task Implementation, Only support lighteval and lmeval.")
    
    # ===============================
    # gather the results into a single csv file
    # ===============================
    results_dir = os.path.join(eval_analysis_dir, "results")
    summary_filename = os.path.join(eval_analysis_dir, args.hf_model_dir.replace("/", "_") + "_summary.csv")

    print(f"Gathering results into {summary_filename}")
    if args.task_impl == "lighteval":
        if args.task_set == "fineweb":
            eval_tasks = [
                "custom|arc:challenge|0",
                "custom|arc:easy|0",
                "custom|commonsense_qa|0",
                "custom|hellaswag|0",
                "custom|mmlu:_average|0",
                "custom|openbookqa|0",
                "custom|piqa|0",
                "custom|siqa|0",
                "custom|winogrande|0",
                "lighteval|sciq|0",
            ]
        else:
            raise ValueError("Unknown Task Set, Only support fineweb.")
    
    elif args.task_impl == "lmeval":
        if args.task_set == "indicator":
            eval_tasks = []
        else:
            raise ValueError("Unknown Task Set, Only support indicator.")
    # write headers
    with open(summary_filename, "a") as f:
        f.write("model_step")
        for task in eval_tasks:
            f.write(f",{task}")
        f.write("\n")

    # recursive find json files
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            """
            extract 5B from the suffix
            e.g., tinylm_hf_pt_redpj_raw_5B
            """ 
            with open(os.path.join(root, file), "r") as f:
                info = json.load(f)
                results = info["results"]
                model_step = int(info["config_general"]["model_name"].split("_")[-1][:-1])            
                # for same model, eval results for different tasks in one row
                with open(summary_filename, "a") as f:
                    f.write(f"{model_step}")
                    for task in eval_tasks:
                        if task in results:
                            f.write(f",{results[task][args.metric]}")
                        else:
                            f.write(",N/A")
                    f.write("\n")
