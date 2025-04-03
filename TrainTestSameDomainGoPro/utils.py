import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.models import vgg19


# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# use for vgg which use the specific mean, std on Imagenet dataset
def normalize_image(image):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    return normalize(image)


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:16].eval().cuda()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # Normalize images
        generated_normalized = normalize_image(generated)
        target_normalized = normalize_image(target)
        # Extract features
        gen_features = self.vgg(generated_normalized)
        target_featuers = self.vgg(target_normalized)
        loss = torch.nn.MSELoss()
        return loss(gen_features, target_featuers)


def plot_loss_and_lr(d_losses, g_losses, d_lrs, g_lrs, num_epochs, save=True, save_dir="results/", show=False):
    fig, ax1 = plt.subplots()

    # Losses
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    if len(d_losses) > 0:
        ax1.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss", color='red')
    ax1.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the learning rates
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(range(1, len(d_lrs) + 1), d_lrs, label="Discriminator LR", color='purple', linestyle='--')
    ax2.plot(range(1, len(g_lrs) + 1), g_lrs, label="Generator LR", color='green', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped

    # Save or show the figure
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = os.path.join(save_dir, f"Training_Loss_LR_values_epoch_{num_epochs}.png")
        if len(d_losses)== 0:
            save_fn = os.path.join(save_dir, f"Validation_Loss_LR_values_epoch_{num_epochs}.png")
        plt.savefig(save_fn)
        print(f"Saved plot to {save_fn}")

    if show:
        plt.show()
    else:
        plt.close()



# Function to unnormalize and scale image
def restore_image(img):
    # Asumming img is a Tesnsor of shape (C.H.W)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    # torch .Size([1, 3, 256, 256])
    img = img[0].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) => (H, W, C)
    img = (img * std + mean) * 255  # Unnormalize and scale to 0-255
    img = np.clip(img, 0, 255).astype(
        np.uint8
    )  # clip values to ensure they are withing [0, 255] and convert to uint8

    return img


def plot_test_result(
    input,
    target,
    gen_image,
    index,
    training=True,
    save=False,
    save_dir="results/",
    show=False,
    fig_size=(15, 5),  # Increase figure size for larger images
    **kwargs,
):
    """
    Expect each image as pytorch tensor
    """
    if not training:
        fig_size = (input.size(2) * 3 / 100, input.size(3) / 100)

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    imgs = [input, gen_image, target]

    for ax, img in zip(axes.flatten(), imgs):
        ax.axis("off")
        ax.set_adjustable("box")
        # Scale to 0-255
        img = restore_image(img)
        ax.imshow(img, aspect="equal")

    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust space between plots

    # Save figure to a temporary file
    temp_save_fn = "temp_result.png"
    plt.savefig(temp_save_fn, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Load image with PIL
    img = Image.open(temp_save_fn)
    width, height = img.size
    padding = 50 + 10 * len(kwargs)  # Add space for text
    new_height = height + padding  # New height includes padding
    new_img = Image.new("RGB", (width, new_height), "white")
    new_img.paste(img, (0, 0))

    # Draw text
    draw = ImageDraw.Draw(new_img)
    metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in kwargs.items()])
    font = ImageFont.load_default()
    text_y = height + 10  # Starting y position for the text
    for line in metrics_text.split("\n"):
        text_width, text_height = draw.textbbox((0, 0), line, font=font)[2:]
        text_x = (width - text_width) / 2
        draw.text((text_x, text_y), line, fill="black", font=font)
        text_y += text_height + 5  # Move to the next line

    # Save the final image
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if training:
            final_save_fn = os.path.join(save_dir, f"Result_epoch_{index + 1}.png")
        else:
            final_save_fn = os.path.join(save_dir, f"Test_result_{index + 1}.png")
        new_img.save(final_save_fn)
    os.remove(temp_save_fn)  # Remove the temporary image file

    if show:
        new_img.show()
    else:
        plt.close()


# Make gif
def make_gif(dataset, num_epochs, save_dir="results/"):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = os.path.join(save_dir, f"Result_epoch_{epoch + 1}.png")
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(
        os.path.join(save_dir, f"{dataset}_pix2pix_epochs_{num_epochs}.gif"),
        gen_image_plots,
        duration=200,
    )


# Make gif
def make_gif(dataset, num_epochs, save_dir="results/"):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + "Result_epoch_{:d}".format(epoch + 1) + ".png"
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(
        save_dir + dataset + "_pix2pix_epochs_{:d}".format(num_epochs) + ".gif",
        gen_image_plots,
        duration=200,
    )
