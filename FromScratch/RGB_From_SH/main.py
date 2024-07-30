import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sh_utils import *

IMAGE_SIZE = 50


def load_from_checkpoint(checkpoint_address, model):
    checkpoint = torch.load(checkpoint_address)
    model.load_state_dict(checkpoint)
    return model


class SHPredictor(nn.Module):
    def __init__(self, input_dim, sh_order):
        """
        Initialize the SHPredictor module.

        Parameters:
        - input_dim: int, the dimension of the input features.
        - sh_order: int, the order of spherical harmonics.
        """
        super(SHPredictor, self).__init__()
        self.sh_order = sh_order
        self.num_sh_coeffs = (sh_order + 1) ** 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * self.num_sh_coeffs),  # 3 is for RGB
        )

    def forward(self, x):
        """
        Forward pass of the SHPredictor.

        Parameters:
        - x: torch.Tensor, the input features, shape (N, input_dim).

        Returns:
        - sh_coeffs: torch.Tensor, the predicted SH coefficients, shape (N, num_sh_coeffs).
        """
        sh_coeffs = self.fc(x)
        sh_coeffs = sh_coeffs.view(-1, 3, self.num_sh_coeffs)
        return sh_coeffs


def render_colors(sh_coeffs, directions, sh_order):
    """
    Render colors from SH coefficients.

    Parameters:
    - sh_coeffs: torch.Tensor, the SH coefficients, shape (N, num_sh_coeffs).
    - directions: torch.Tensor, the viewing directions, shape (N, 3).
    - sh_order: int, the order of spherical harmonics.

    Returns:
    - colors: torch.Tensor, the rendered colors, shape (N, 3).
    """
    evaled_sh_colors = eval_sh(sh_order, sh_coeffs, directions)
    return SH2RGB(evaled_sh_colors)


# Example training loop
def train(
    model,
    optimizer,
    input_features,
    ground_truth_colors,
    directions,
    sh_order,
    num_epochs,
):
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Predict SH coefficients
        sh_coeffs = model(input_features)

        # Render colors
        rendered_colors = render_colors(sh_coeffs, directions, sh_order)

        # Compute loss
        loss = criterion(rendered_colors, ground_truth_colors)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}")
            model.eval()
            rendered_colors = render_colors(model(input_features), directions, sh_order)
            plot_images(ground_truth_colors, rendered_colors)


def plot_images(ground_truth, rendered):
    """
    Plot the ground truth and rendered images.

    Parameters:
    - ground_truth: torch.Tensor, the ground truth colors, shape (N, 3).
    - rendered: torch.Tensor, the rendered colors, shape (N, 3).
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imsave(
        "ground_truth.png",
        ground_truth.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        .cpu()
        .detach()
        .numpy()
        .clip(0, 1),
    )

    plt.subplot(1, 2, 2)
    plt.title("Rendered")
    plt.imsave(
        "rendered.png",
        rendered.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).cpu().detach().numpy().clip(0, 1),
    )
    plt.close("all")


input_dim = 3
sh_order = 2
model = SHPredictor(input_dim, sh_order)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

target_image = Image.open("bunny.jpg")
target_image = target_image.resize((IMAGE_SIZE, IMAGE_SIZE))
target_colors = torch.tensor(np.array(target_image).reshape(-1, 3) / 255.0).float()

# Assuming random input features (XYZ coordinates) and directions (unit vectors)
input_features = torch.rand((IMAGE_SIZE * IMAGE_SIZE, input_dim))
directions = torch.rand((IMAGE_SIZE * IMAGE_SIZE, 3))
directions = directions / torch.norm(directions, dim=1, keepdim=True) # Normalize directions

num_epochs = 10000
train(model, optimizer, input_features, target_colors, directions, sh_order, num_epochs)
