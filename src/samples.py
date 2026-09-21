import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.scheduler import NoiseScheduler
from src.unet import Unet

PATH = "models/ddpm_cifar10_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = Unet(in_channels=3, out_channel=128, kernel_size=3, time_dim=128).to(DEVICE)
SCHEDULER = NoiseScheduler()

def load_model(model, path): return model.load_state_dict(torch.load(PATH, map_location=DEVICE))

def _sample(model, noise_scheduler, n_samples, device):
    model.eval()
    betas = noise_scheduler.betas
    alphas = noise_scheduler.alphas
    alpha_bars = noise_scheduler.alpha_bars
    with torch.no_grad():
        xt = torch.randn(n_samples, 1, 28, 28).to(device)   
        for t in range(999, 0, -1):
            t_batch = torch.tensor([t] * n_samples).to(device)
            pred_noise = model(xt, t_batch)
            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            sigma_t = torch.sqrt(betas[t])
            x_prev = (1/torch.sqrt(alphas[t])) * (xt - (betas[t]/torch.sqrt(1 - alpha_bars[t])) * pred_noise) + sigma_t * z
            xt = x_prev
    return xt

def _plot(samples, filename:str):
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i, ax in enumerate(axes):
        img = samples[i, 0].cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.savefig(filename, dpi=150)
    plt.show()

def _sample_cifar10(model, noise_scheduler, n_samples, device):
    model.eval()
    betas = noise_scheduler.betas.to(device)
    alphas = noise_scheduler.alphas.to(device)
    alpha_bars = noise_scheduler.alpha_bars.to(device)
    with torch.no_grad():
        xt = torch.randn(n_samples, 3, 32, 32).to(device)   
        for t in range(999, -1, -1):
            t_batch = torch.tensor([t] * n_samples).to(device)
            pred_noise = model(xt, t_batch)
            pred_nosie = pred_noise.clamp(-1, 1)
            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            sigma_t = torch.sqrt(betas[t])
            denom  = torch.sqrt(alphas[t]) + 1e-8
            x_prev = (1/denom) * (xt - (betas[t]/torch.sqrt(1 - alpha_bars[t] + 1e-8)) * pred_noise) + sigma_t * z
            xt = x_prev.clamp(-2, 2)
    return xt

def _plot(samples, filename):
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i, ax in enumerate(axes):
        img = samples[i].cpu().numpy()  # (3, 32, 32)
        img = img.transpose(1, 2, 0)   # (32, 32, 3)
        img = (img + 1) / 2            # denormalize from [-1,1] to [0,1]
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == '__main__':
    load_model(model=MODEL, path=PATH)
    _plot(model=MODEL, scheduler=SCHEDULER, n_samples=8, device=DEVICE)