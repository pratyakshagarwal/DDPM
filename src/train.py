import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from src.scheduler import NoiseScheduler
from src.unet import Unet
from src.samples import _plot, _sample, _sample_cifar10

def train(model, noise_scheduler, train_dl, val_dl, optimizer,
           epochs, device, lr_scheduler, n_samples:int=8, _sample:callable=_sample, model_name:str="ddpm"):
    print(f"Using: {device}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # training loop
        model.train()
        train_loss = 0
        for x0, _ in train_dl:
            x0 = x0.to(device)
            t = torch.randint(0, 1000, size=(x0.shape[0],)).to(device)
            true_noise = torch.randn_like(x0)
            noisy_x0 = noise_scheduler.add_noise(x0, t, true_noise)
            pred_noise = model(noisy_x0, t)
            _mse = F.mse_loss(pred_noise, true_noise)
            _mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        
            optimizer.zero_grad()
            train_loss += _mse.item()

        
        # validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x0, _ in val_dl:
                x0 = x0.to(device)
                t = torch.randint(0, 1000, size=(x0.shape[0],)).to(device)
                true_noise = torch.randn_like(x0)
                noisy_x0 = noise_scheduler.add_noise(x0, t, true_noise)
                pred_noise = model(noisy_x0, t)
                _mse = F.mse_loss(pred_noise, true_noise)
                val_loss += _mse.item()

        if epoch % 10 == 0:
            samples = _sample(model, noise_scheduler, n_samples=n_samples, device=device)
            filesname = "plots/" + model_name + f"samples{epoch}.png"
            _plot(samples, filesname)
        
        # checkpointing
        avg_val = val_loss / len(val_dl)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f'models/{model_name}.pth')
            print(f"Saved best model at epoch {epoch}")
        
        print(f"Epoch {epoch} | Train: {train_loss/len(train_dl):.4f} | Val: {avg_val:.4f}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_scheduler = NoiseScheduler()
    ## Fashion Mnist Training 
    batch_size=64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))  # normalize to [-1, 1]
    ])

    dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lr = 2e-4
    epochs = 70
    steps_per_epoch = len(dl) 

    model = Unet(in_channels=1, out_channel=64, kernel_size=3, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    train(model, noise_scheduler, dl, val_dl, optimizer=optimizer, epochs=epochs, device=device, lr_scheduler=scheduler,
        _sample=_sample, model_name="ddpm_fashionmnist")

    ## Cifar10 Training 
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
])

    dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Unet(in_channels=3, out_channel=128, kernel_size=3, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"no of model parameters: sum([p.numel() for p in model.parameters()])")
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=steps_per_epoch)

    train(model, noise_scheduler, dl, val_dl, optimizer, epochs, device, lr_scheduler=lr_scheduler, 
        _sample=_sample_cifar10, model_name="ddpm_cifar10")