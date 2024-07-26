import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from ddpm import DDPMSampler
from config import WIDTH, HEIGHT, LATENTS_WIDTH, LATENTS_HEIGHT
from dataloaders import  TrafficSignsDataset, load_data, ReshapeTransform
from utils import rescale, get_time_embedding


def train(
    dataset_path,
    prompt,
    uncond_prompt=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    epochs=10,
    batch_size=4,
    lr=1e-4,
    size=(128, 128),
):
    
    # Prepare dataset and dataloader

    images, labels = load_data(dataset_path)
    print(f"Number of images: {len(images)}")
    print(f"Shape of the first image: {images[0].shape}")
    # Create DataLoader
    dataset = TrafficSignsDataset(images, transform=True, size=size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Number of batches: {len(dataloader)}")
    print(f"Shape of the first batch: {next(iter(dataloader))[0].shape}")

    # Initialize models
    encoder = models["encoder"].to(device)
    diffusion = models["diffusion"].to(device)
    decoder = models["decoder"].to(device)
    clip = models["clip"].to(device)

    # Set models to training mode
    encoder.train()
    diffusion.train()
    decoder.train()
    clip.train()

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = Adam(list(encoder.parameters()) + list(diffusion.parameters()) + list(decoder.parameters()), lr=lr)

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Encode images
            encoder_noise = torch.randn(batch.shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH, generator=generator, device=device)
            latents = encoder(batch, encoder_noise)

            # Add noise to the latents (the encoded input image)
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
            sampler.set_strength(strength=strength)
            noisy_latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Generate context
            if do_cfg:
                cond_tokens = tokenizer.batch_encode_plus([prompt] * batch.shape[0], padding="max_length", max_length=77).input_ids
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
                cond_context = clip(cond_tokens)
                uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt] * batch.shape[0], padding="max_length", max_length=77).input_ids
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
                uncond_context = clip(uncond_tokens)
                context = torch.cat([cond_context, uncond_context])
            else:
                tokens = tokenizer.batch_encode_plus([prompt] * batch.shape[0], padding="max_length", max_length=77).input_ids
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                context = clip(tokens)

            # Diffusion steps
            timesteps = sampler.timesteps
            for timestep in timesteps:
                time_embedding = get_time_embedding(timestep).to(device)
                model_input = noisy_latents.repeat(2, 1, 1, 1) if do_cfg else noisy_latents
                model_output = diffusion(model_input, context, time_embedding)
                if do_cfg:
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                noisy_latents = sampler.step(timestep, noisy_latents, model_output)

            # Decode latents
            recon_images = decoder(noisy_latents)

            # Calculate loss and update model parameters
            loss = criterion(recon_images, batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save trained models
    # model_file = "../data/v1-5-pruned-emaonly.ckpt"
    # models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
    model_save_path = "../data/finetuned_models.ckpt"
    torch.save({
        "encoder": encoder.state_dict(),
        "diffusion": diffusion.state_dict(),
        "decoder": decoder.state_dict(),
        "clip": clip.state_dict()
    }, model_save_path)
    print(f"Models saved to {model_save_path}")
    


