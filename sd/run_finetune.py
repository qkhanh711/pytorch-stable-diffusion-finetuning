# %%
import torch 
print(torch.cuda.device_count())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import model_loader
from PIL import Image
from transformers import CLIPTokenizer


DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

prompt = "A trafic sign on a beautiful beach."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14
image_path = "../images/dog.png"
input_image = Image.open(image_path)
strength = 0.48


sampler = "ddpm"
num_inference_steps = 6
seed = 43

# %%
from finetune import train, TrafficSignTrainer
import pytorch_lightning as pl

# train(
#     dataset_path="../../Training",
#     prompt=prompt,
#     uncond_prompt=uncond_prompt,
#     strength=strength,
#     do_cfg=do_cfg,
#     cfg_scale=cfg_scale,
#     sampler_name=sampler,
#     n_inference_steps=num_inference_steps,
#     seed=seed,
#     models=models,
#     device=DEVICE,
#     idle_device="cuda",
#     tokenizer=tokenizer,
#     epochs=1,
#     batch_size=1,
#     lr=1e-4,
#     size=(512,512),
# )
print(DEVICE)
print(type(DEVICE))
# Use this class in your training script
trainer = pl.Trainer(max_epochs=10, gpus=1)
traffic_sign_trainer = TrafficSignTrainer(
    models=models,
    tokenizer=tokenizer,
    dataset_path="../../Sample",
    prompt=prompt,
    uncond_prompt= uncond_prompt,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale= cfg_scale,
    sampler_name= sampler,
    device=DEVICE,
    n_inference_steps= num_inference_steps,
    batch_size=1,
    lr=1e-4,
    size=(512, 512),
    seed=42
)

trainer.fit(traffic_sign_trainer)

