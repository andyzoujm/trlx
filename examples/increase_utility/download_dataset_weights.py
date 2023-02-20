import os
import argparse
import wget
import tarfile
import gdown

ETHICS_DATASET = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"
UTILITY_WEIGHTS = "https://drive.google.com/uc?id=1MHvSFbHjvzebib90wW378VtDAtn1WVxc"

def download_dataset(url: str, output: str):
    wget.download(url, out=output)

def extract_tar(tar_path: str):
    tar = tarfile.open(tar_path)
    tar.extractall()
    tar.close()

def download_weights(url: str, output: str):
    gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_checkpoint_path", type=str, default="reward_model/util_roberta-large.pt", help="path to save reward model weights")
    parser.add_argument("--dataset_path", type=str, default="ethics.tar", help="path to save tar file for ETHICSs")
    args = parser.parse_args()
    
    dataset_output = os.path.join(os.getcwd(), args.dataset_path)
    weights_output = os.path.join(os.getcwd(), args.reward_checkpoint_path)
    os.makedirs("reward_model", exist_ok=True)
    download_dataset(ETHICS_DATASET, dataset_output)
    print("ETHICS dataset downloaded")
    extract_tar(args.dataset_path)
    print("ETHICS dataset extracted")
    download_weights(UTILITY_WEIGHTS, weights_output)
    print("Utility weights downloaded")
