import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sh_utils import *
import imageio
import os

IMAGE_SIZE = 256


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


def append_to_gif(image_path, gif_path):
    """
    Append a new image to the GIF.

    Parameters:
    - image_path: str, the path to the new image.
    - gif_path: str, the path to the GIF file.
    """
    new_image = imageio.imread(image_path)[..., :3]
    if os.path.exists(gif_path):
        existing_gif = imageio.mimread(gif_path)
        existing_gif.append(new_image)
        imageio.mimsave(gif_path, existing_gif, duration=0.5)
    else:
        imageio.mimsave(gif_path, [new_image], duration=0.5)


def plot_images(ground_truth, rendered, image_path):
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
        image_path,
        rendered.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).cpu().detach().numpy().clip(0, 1),
    )
    plt.close("all")


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

        sh_coeffs = model(input_features)

        rendered_colors = render_colors(sh_coeffs, directions, sh_order)

        loss = criterion(rendered_colors, ground_truth_colors)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.7f}")
            model.eval()
            rendered_colors = render_colors(model(input_features), directions, sh_order)

            if not os.path.exists("renders"):
                os.makedirs("renders", exist_ok=True)
            image_path = f"renders/rendered_epoch_{epoch + 1}.png"
            plot_images(ground_truth_colors, rendered_colors, image_path)
            append_to_gif(image_path, "training_process.gif")

            torch.save(model.state_dict(), "model.pth")


input_dim = 2
sh_order = 3
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = SHPredictor(input_dim, sh_order)

if os.path.exists("model.pth"):
    model = load_from_checkpoint("model.pth", model)
    print("Model loaded from checkpoint.")

model.to(device)
LR = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

target_image = Image.open("bunny.jpg")
target_image = target_image.resize((IMAGE_SIZE, IMAGE_SIZE))
target_colors = torch.tensor(np.array(target_image).reshape(-1, 3) / 255.0).float()
target_colors = target_colors.to(device)

x, y = np.meshgrid(
    np.linspace(-1, 1, IMAGE_SIZE), np.linspace(-1, 1, IMAGE_SIZE), indexing="ij"
)

x = x.reshape(-1)
y = y.reshape(-1)

input_features = torch.tensor(np.stack([x, y], axis=1)).float()
input_features = input_features.to(device)

directions = torch.rand((IMAGE_SIZE * IMAGE_SIZE, 3)).to(device)
directions = directions / torch.norm(directions, dim=1, keepdim=True).to(
    device
)  # Normalize directions

num_epochs = 3000
train(model, optimizer, input_features, target_colors, directions, sh_order, num_epochs)
