import argparse
from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil


def download_and_save_model(repo_id, excluded_files, save_dir, download_f16):
    try:
        # List all files in the repository
        repo_files = list_repo_files(repo_id)

        # Ensure the target directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Loop through all files in the repository
        for filename in repo_files:
            # Skip the excluded file if download_f16 is False
            if filename not in excluded_files or (filename == "Meta-Llama-3.1-8B-Instruct-F16.gguf" and download_f16):
                print(f"Downloading {filename}...")
                downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename)

                # Define the save path for each file
                save_path = os.path.join(save_dir, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Copy the file to the target directory
                shutil.copy(downloaded_file, save_path)
                print(f"Saved {filename} to {save_path}")
            else:
                print(f"Skipping {filename} (excluded).")
    except Exception as e:
        print(f"An error occurred: {e}")


# Set up argument parser
parser = argparse.ArgumentParser(description="Download specific model files.")
parser.add_argument(
    "--dl-f16", action="store_true", help="Set this flag to download Meta-Llama-3.1-8B-Instruct-F16.gguf"
)
parser.add_argument(
    "--repo-id", type=str, default="trinhvanhung/Meta-Llama-3.1-8B-Instruct-Q4_K_M", help="Hugging Face model repository ID"
)
parser.add_argument(
    "--save-dir", type=str, default="./Meta-Llama-3.1-8B-Instruct-Q4_K_M", help="Directory to save downloaded files"
)

args = parser.parse_args()

# Call the function based on command-line arguments
download_and_save_model(
    repo_id=args.repo_id,
    excluded_files=["Meta-Llama-3.1-8B-Instruct-F16.gguf"],  # Default exclude F16 file
    save_dir=args.save_dir,
    download_f16=args.dl_f16  # Download the F16 file if the flag is set
)
