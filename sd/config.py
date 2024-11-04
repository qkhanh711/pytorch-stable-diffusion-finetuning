import torch
import json

with open("config.json", "r") as f:
    args = json.load(f)

WIDTH = args["size"][0]
HEIGHT = args["size"][1]
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def check_device(device, allow_cuda=True, allow_mps=False):
    if torch.cuda.is_available() and allow_cuda:
        return "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and allow_mps:
        return "mps"
    return "cpu"