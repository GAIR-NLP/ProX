import argparse


def main(args):
    model_file_path = args.model_path
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_file_path)
    model.save_pretrained(model_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    main(args)
