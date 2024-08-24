import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TimeEmbedding(nn.Module):
    def __init__(self, n_steps, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_steps, embedding_dim)

    def forward(self, t):
        return self.embedding(
            t.to(self.embedding.weight.device)
        )  # Ensure t is on the same device as the embedding weights


class EpsModel(nn.Module):
    def __init__(self, base_model, time_embedding_dim):
        super(EpsModel, self).__init__()
        self.base_model = base_model
        self.time_embedding_dim = time_embedding_dim
        self.time_embedding = TimeEmbedding(1000, time_embedding_dim)
        self.fc = nn.Linear(
            time_embedding_dim, 32 * 32 * 1
        )  # Adjust to 1 channel for time embedding

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        t_emb = self.fc(t_emb.view(-1, self.time_embedding_dim)).view(-1, 1, 32, 32)
        x = torch.cat([x, t_emb], dim=1)
        return self.base_model(x)


class DenoiseDiffusion(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int):
        super(DenoiseDiffusion, self).__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to("mps")
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to("mps")
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean = x0 * (self.alpha_bar[t].to(x0.device) ** 0.5)
        variance = 1.0 - self.alpha_bar[t].to(x0.device)
        return mean, variance

    def sample_q(self, x0: torch.Tensor, t: torch.Tensor, eps=None) -> torch.Tensor:
        mean, variance = self.q_xt_x0(x0, t)
        if eps is None:
            eps = torch.randn_like(x0)
        return mean + eps * (variance**0.5)

    def sample_p(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_theta = self.eps_model(xt, t).view(xt.shape)
        eps_coeff = (1.0 - self.alpha[t]) / (1.0 - self.alpha_bar[t]) ** 0.5
        mean = 1.0 / (self.alpha[t] ** 0.5) * (xt - eps_coeff * eps_theta)
        eps = torch.randn_like(xt)
        return mean + eps * (self.sigma2[t] ** 0.5)

    def loss(self, x0: torch.Tensor):
        batch_size = x0.shape[0]
        t = torch.randint(
            0, self.n_steps, (batch_size,), dtype=torch.long, device=x0.device
        )

        noise = torch.randn_like(x0)

        xt = self.sample_q(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t).view(batch_size, -1, 32, 32)
        return torch.functional.F.mse_loss(noise, eps_theta)


def generate_image(model, n_steps, device):
    model.eval()
    with torch.no_grad():
        xt = torch.randn(1, 1, 32, 32, device=device)  # Start with a random noise image
        for step in reversed(range(n_steps)):
            t = torch.tensor([step], device=device, dtype=torch.long)
            xt = model.sample_p(xt, t)
        generated_image = xt.squeeze().cpu().numpy()
        generated_image = np.clip(generated_image, 0, 1)
    return generated_image


if __name__ == "__main__":
    base_model = resnet18(pretrained=True)
    base_model.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=2, padding=3, bias=False
    )  # Adjust input channels to 4
    base_model.fc = nn.Linear(base_model.fc.in_features, 1 * 32 * 32)
    base_model = base_model.to("mps")

    eps_model = EpsModel(base_model, time_embedding_dim=128)

    diffusion = DenoiseDiffusion(eps_model, 1000)
    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=Compose([Resize((32, 32)), ToTensor()]),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    device = torch.device("mps")
    diffusion.to(device)
    eps_model.to(device)
    optimizer = torch.optim.Adam(diffusion.parameters())

    NUM_EPOCHS = 10000
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        for i, (x, _) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            loss = diffusion.loss(x)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": loss.item()})

        if epoch % 50 == 0:
            generated_image = generate_image(diffusion, 1000, device)
            plt.imshow(generated_image, cmap="gray")
            plt.axis("off")
            plt.imsave(f"generated_image_{i}.png", generated_image, cmap="gray")
            plt.close()
