import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from ddpm import DDPMSampler
from config import WIDTH, HEIGHT, LATENTS_WIDTH, LATENTS_HEIGHT
from dataloaders import  TrafficSignsDataset, load_data, ReshapeTransform
from utils import rescale, get_time_embedding
import pytorch_lightning as pl


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
    parallel=False,
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
    model_save_path = "../../data/finetuned_models.ckpt"
    torch.save({
        "encoder": encoder.state_dict(),
        "diffusion": diffusion.state_dict(),
        "decoder": decoder.state_dict(),
        "clip": clip.state_dict()
    }, model_save_path)
    print(f"Models saved to {model_save_path}")
    

class TrafficSignTrainer(pl.LightningModule):
    def __init__(self, models, tokenizer, dataset_path, prompt, uncond_prompt=None, strength=0.8, do_cfg=True, cfg_scale=7.5,
                 sampler_name="ddpm", n_inference_steps=50, batch_size=4, lr=1e-4, size=(128, 128), seed=None, device="cuda"):
        super(TrafficSignTrainer, self).__init__()
        self.encoder = models["encoder"]
        self.diffusion = models["diffusion"]
        self.decoder = models["decoder"]
        self.clip = models["clip"]
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.prompt = prompt
        self.uncond_prompt = uncond_prompt
        self.strength = strength
        self.do_cfg = do_cfg
        self.cfg_scale = cfg_scale
        self.sampler_name = sampler_name
        self.n_inference_steps = n_inference_steps
        self.batch_size = batch_size
        self.lr = lr
        self.size = size
        self.seed = seed
        self._device = torch.device(device)
        self.automatic_optimization = False  # Disable automatic optimization

        # Prepare dataset and dataloader
        images, _ = load_data(dataset_path)
        self.dataset = TrafficSignsDataset(images, transform=True, size=size)
        self.criterion = torch.nn.MSELoss()

        # Initialize random number generator according to the seed specified
        self.generator = torch.Generator(device=self._device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        return Adam(list(self.encoder.parameters()) + list(self.diffusion.parameters()) + list(self.decoder.parameters()), lr=self.lr)

    def training_step(self, batch, batch_idx):
        batch = batch.to(self._device)
        self.encoder.train()
        self.diffusion.train()
        self.decoder.train()
        self.clip.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # Encode images
        encoder_noise = torch.randn(batch.shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH, generator=self.generator, device=self._device)
        latents = self.encoder(batch, encoder_noise)

        # Add noise to the latents (the encoded input image)
        sampler = DDPMSampler(self.generator)
        sampler.set_inference_timesteps(self.n_inference_steps)
        sampler.set_strength(strength=self.strength)
        noisy_latents = sampler.add_noise(latents, sampler.timesteps[0])

        # Generate context
        if self.do_cfg:
            cond_tokens = self.tokenizer.batch_encode_plus([self.prompt] * batch.shape[0], padding="max_length", max_length=77).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self._device)
            cond_context = self.clip(cond_tokens)
            uncond_tokens = self.tokenizer.batch_encode_plus([self.uncond_prompt] * batch.shape[0], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=self._device)
            uncond_context = self.clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = self.tokenizer.batch_encode_plus([self.prompt] * batch.shape[0], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=self._device)
            context = self.clip(tokens)

        # Diffusion steps
        timesteps = sampler.timesteps
        for timestep in timesteps:
            time_embedding = get_time_embedding(timestep).to(self._device)
            model_input = noisy_latents.repeat(2, 1, 1, 1) if self.do_cfg else noisy_latents
            model_output = self.diffusion(model_input, context, time_embedding)
            if self.do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = self.cfg_scale * (output_cond - output_uncond) + output_uncond
            noisy_latents = sampler.step(timestep, noisy_latents, model_output)

        # Decode latents
        recon_images = self.decoder(noisy_latents)

        # Calculate loss and update model parameters
        loss = self.criterion(recon_images, batch)
        self.manual_backward(loss)
        optimizer.step()

        self.log('train_loss', loss)
        return loss

    def on_epoch_end(self):
        print(f'Epoch {self.current_epoch + 1} completed')

