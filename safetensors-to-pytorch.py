import argparse
from safetensors.torch import load_file
import torch
import os

parser = argparse.ArgumentParser(description="Convert Safetensors to PyTorch models")
parser.add_argument("base_model_path", type=str, help="Base path to the model directory")
args = parser.parse_args()

for i in ["", "_1"]:
    safetensors_model_path = os.path.join(args.base_model_path, f"model{i}.safetensors")
    pytorch_model_path = os.path.join(args.base_model_path, f"pytorch_model{i}.bin")
    torch.save(load_file(safetensors_model_path), pytorch_model_path)