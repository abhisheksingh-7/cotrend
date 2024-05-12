import pandas as pd
import json
import os
import random
from loguru import logger


def ensure_directory(directory):
    """Ensure the directory exists, if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory {directory} created.")


def reduce_file_size(
    source_directory,
    file_name,
    output_directory,
    reduction_factor_json=0.1,
    reduction_factor_tsv=0.001,
):
    file_path = os.path.join(source_directory, file_name)
    _, file_extension = os.path.splitext(file_name)

    # Process JSON files
    if file_extension == ".json":
        with open(file_path, "r") as file:
            data = json.load(file)

        # Check if data is a list and perform sampling
        if isinstance(data, list):
            sample_size = int(len(data) * reduction_factor_json)
            reduced_data = random.sample(data, sample_size)
            logger.info(f"Sampled {sample_size} entries from {file_name}.")
        else:
            reduced_data = data  # Handle non-list data structures if necessary
            logger.warning(f"No sampling performed on {file_name} as it's not a list.")

        output_file_path = os.path.join(output_directory, f"reduced_{file_name}")
        with open(output_file_path, "w") as file:
            json.dump(reduced_data, file, indent=4)
        logger.info(f"Reduced JSON file saved as {output_file_path}.")

    # Process TSV files
    elif file_extension == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
        reduced_df = df.sample(frac=reduction_factor_tsv)  # Random sample of rows
        output_file_path = os.path.join(output_directory, f"reduced_{file_name}")
        reduced_df.to_csv(output_file_path, sep="\t", index=False)
        logger.info(f"Reduced TSV file saved as {output_file_path}.")


# Set up logging
logger.add("data_reduction.log", rotation="10 MB")

# Directories and files setup
base_directory = "/data/clapt/eval/fid"
subdirectories = ["NQ", "TQA"]
files = ["psgs_w100.tsv"]
output_base = os.getcwd()  # Use the current working directory as the base for outputs

# Ensure output directories exist
output_dirs = {
    subdir: os.path.join(output_base, subdir) for subdir in subdirectories + ["Wiki"]
}
for dir_path in output_dirs.values():
    ensure_directory(dir_path)

# Reduce JSON files in subdirectories
for subdirectory in subdirectories:
    subdirectory_path = os.path.join(base_directory, subdirectory)
    output_dir = output_dirs[subdirectory]
    for file_name in os.listdir(subdirectory_path):
        logger.info(f"Processing {file_name} in {subdirectory_path}.")
        reduce_file_size(
            subdirectory_path, file_name, output_dir, reduction_factor_json=0.1
        )

# Reduce the TSV file and assume it belongs to 'Wiki'
logger.info(f"Processing {files[0]} in {base_directory}.")
reduce_file_size(
    base_directory, files[0], output_dirs["Wiki"], reduction_factor_tsv=0.001
)
