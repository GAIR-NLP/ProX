import argparse
import glob
import os
import random
import shutil


def main(args: argparse.Namespace):
    # Get all json.gz files
    json_files = glob.glob(os.path.join(args.input_dir, "*.json.gz"))

    # Shuffle the files
    random.shuffle(json_files)

    # Create dump directories if they don't exist
    for i in range(1, args.num_dumps + 1):
        dump_dir = os.path.join(args.output_base_dir, f"dump_{i}")
        os.makedirs(dump_dir, exist_ok=True)

    # Distribute files across dump directories
    for i, file_path in enumerate(json_files):
        dump_index = i % args.num_dumps + 1
        destination = os.path.join(
            args.output_base_dir, f"dump_{dump_index}", os.path.basename(file_path)
        )

        # Move the file
        shutil.move(file_path, destination)
        print(f"Moved {file_path} to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshuffle data")
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_base_dir", type=str, help="Output base directory")
    parser.add_argument("--num_dumps", type=int, help="Number of dump directories")
    args = parser.parse_args()
    main(args)
