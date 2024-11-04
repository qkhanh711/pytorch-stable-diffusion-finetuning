from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from torch import nn
import torch
import model_converter

def parallelize_model(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
        
    else:
        model = model
    return model

def new_state_dict(state_dict):
    new_state_dict = state_dict.copy()
    for key in state_dict.keys():
        keys = "module." + key
        new_state_dict[keys] = state_dict[key]
        del new_state_dict[key]
    return new_state_dict

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder = parallelize_model(encoder)
    state_dict['encoder'] = new_state_dict(state_dict['encoder'])
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder = parallelize_model(decoder)
    state_dict["decoder"] = new_state_dict(state_dict["decoder"])
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion = parallelize_model(diffusion)
    state_dict['diffusion'] = new_state_dict(state_dict['diffusion'])
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip = parallelize_model(clip)
    state_dict['clip'] = new_state_dict(state_dict['clip'])
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }