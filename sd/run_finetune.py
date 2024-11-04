import torch
import model_loader
from parallel_model_loader import preload_models_from_standard_weights
from PIL import Image
from transformers import CLIPTokenizer
from config import HEIGHT, WIDTH
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
from finetune import train, TrafficSignTrainer
import pytorch_lightning as pl
from config import HEIGHT, WIDTH, check_device
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--dataset_path", type=str, default="../../Training")
args.add_argument("--epochs", type=int, default=1)
args.add_argument("--device", type=str, default="cuda")
args.add_argument("--allow_cuda", type=bool, default=True)
args.add_argument("--allow_mpls", type=bool, default=False)
args.add_argument("--lightning_train", type=bool, default=True)
args.add_argument("--num_inference_steps", type=int, default=6)
args.add_argument("--rate", type=float, default=1.0)
args.add_argument("--saved_path", type=str, default="../data/finetuned_models_")
args.add_argument("--prompt", type=str, default="A trafic sign on a beautiful beach.")
args.add_argument("--uncond_prompt", type=str, default="")
args.add_argument("--do_cfg", type=bool, default=True)
args.add_argument("--cfg_scale", type=int, default=8)
args.add_argument("--sampler", type=str, default="ddpm")
args.add_argument("--seed", type=int, default=43)
args.add_argument("--image_path", type=str, default="../images/dog.png")
args.add_argument("--strength", type=float, default=0.48)
args.add_argument("--batch_size", type=int, default=1)
args.add_argument("--lr", type=float, default=1e-4)
args.add_argument("--phase", type=str, default="train")
args.add_argument("--size", type=tuple, default=(512, 512))
args = args.parse_args()

#  save config to json...
import json
with open("config.json", "w") as f:
    json.dump(vars(args), f)

ALLOW_CUDA = True
ALLOW_MPS = False
PARALLEL = False

DEVICE = check_device(args.device, allow_cuda=args.allow_cuda, allow_mps=args.allow_mpls)
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"

if PARALLEL:
    models = preload_models_from_standard_weights(model_file, DEVICE)
else:
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

prompt = "A trafic sign on a beautiful beach."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = False
cfg_scale = 8  # min: 1, max: 14
image_path = "../images/dog.png"
input_image = Image.open(image_path)
strength = 0.48


if args.phase == "train":
    print("Training on {} dataset ...".format(args.dataset_path))
    if args.lightning_train:
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=1)
        traffic_sign_trainer = TrafficSignTrainer(
            models=models,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            prompt=args.prompt,
            uncond_prompt= args.uncond_prompt,
            strength=args.strength,
            do_cfg=args.do_cfg,
            cfg_scale= args.cfg_scale,
            sampler_name= args.sampler,
            device=DEVICE,
            n_inference_steps= args.num_inference_steps,
            batch_size=1,
            lr=args.lr,
            size=(HEIGHT, WIDTH),
            seed=args.seed,
            rate=args.rate,
            path=args.saved_path,
        )

        trainer.fit(traffic_sign_trainer)
        trainer.save_checkpoint(f"{args.saved_path}{args.rate}.ckpt")
    else:
        train(
            dataset_path=args.dataset_path,
            prompt=args.prompt,
            uncond_prompt=args.uncond_prompt,
            strength=args.strength,
            do_cfg=args.do_cfg,
            cfg_scale=args.cfg_scale,
            sampler_name=args.sampler,
            n_inference_steps=args.num_inference_steps,
            seed=args.seed,
            models=models,
            device=DEVICE,
            idle_device="cuda",
            tokenizer=tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            size=(HEIGHT, WIDTH),
            path=args.saved_path,
            rate=args.rate,
        )
    print("Model saved to {}{}.ckpt".format(args.saved_path, args.rate))
else:
    print("Generating image ...")
    model_file = f"{args.saved_path}{args.rate}.ckpt"
    model = torch.load(model_file, map_location=DEVICE)

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
#     size=(HEIGHT, WIDTH),
#     parallel=PARALLEL
# )

if __name__ == "__main__":
    sampler = args.sampler
    num_inference_steps = args.num_inference_steps
    # Define the TrafficSignTrainer
    traffic_sign_trainer = TrafficSignTrainer(
        models=models,
        tokenizer=tokenizer,
        dataset_path="../../Training",
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        device=DEVICE,
        n_inference_steps=num_inference_steps,
        batch_size=1,
        lr=1e-4,
        size=(WIDTH, HEIGHT),
        seed=42
    )

    # Set up PyTorch Lightning Trainer for multi-GPU
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",  # Set to 'gpu' for GPU acceleration
        devices=1,  # Specify the number of GPUs to use, e.g., 2
    )

    # Train the model
    trainer.fit(traffic_sign_trainer)


    # output_image = pipeline.generate(
    #     prompt=args.prompt,
    #     uncond_prompt=args.uncond_prompt,
    #     input_image=Image.open(args.image_path),
    #     strength=args.strength,
    #     do_cfg=args.do_cfg,
    #     cfg_scale=args.cfg_scale,
    #     sampler_name=args.sampler,
    #     n_inference_steps=args.num_inference_steps,
    #     seed=args.seed,
    #     models=models,
    #     device=DEVICE,
    #     idle_device="cpu",
    #     tokenizer=tokenizer,
    # )

    # # Combine the input image and the output image into a single image.
    # output_image = Image.fromarray(output_image)
    # # Save generated image to disk.
    # output_image.save(f"genimage_output_{args.rate}.png")

