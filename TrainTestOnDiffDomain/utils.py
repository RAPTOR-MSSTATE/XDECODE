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


def controlled_exponential_curriculum_odd(current_epoch, start_blur=3, max_blur=29, base_growth_rate=1.15, scale_factor=6):
    """
    Generates an exponentially increasing blur level sequence while ensuring only odd blur levels.
    This version controls the growth rate so that the max blur level is reached after epoch 20 or later.
    This is particular step-wise exponential increase in blur level.

    :param current_epoch: Current epoch while training
    :param start_blur: Initial blur level (must be odd)
    :param max_blur: Maximum blur level (must be odd)
    :param base_growth_rate: The base exponential growth rate (lower value delays reaching max blur)
    :param scale_factor: Adjusts the speed of growth (higher value slows early growth)
    :return: blur level
    """

    if current_epoch == 0:
        blur_level = start_blur  # Start at initial blur level
    else:
        # Apply exponential growth with a scaling factor
        blur_level = start_blur * (base_growth_rate ** (current_epoch / scale_factor))

    blur_level = int(2 * np.floor(blur_level / 2) + 1)  # Ensure the value is odd

    if blur_level > max_blur:
        blur_level = max_blur  # Cap at max blur level


    return blur_level


def sigmoid_curriculum_odd(curr_epoch, total_epochs=150, start_blur=3, max_blur=29, k=0.1, m=50):
    """
    Generates a blur level progression using a modified sigmoid function while ensuring only odd values.

    :param total_epochs: Total number of epochs for training
    :param start_blur: Initial blur level (must be odd)
    :param max_blur: Maximum blur level (must be odd)
    :param k: Steepness of the sigmoid curve (higher k = sharper increase)
    :param m: Midpoint of the sigmoid curve (controls when the increase accelerates)
    :return: blur_level
    """
  
    # Compute sigmoid-based blur level
    blur_level = max_blur / (1 + np.exp(-k * (curr_epoch - m)))
    
    # Convert to an odd integer
    blur_level = int(2 * np.floor(blur_level / 2) + 1)
    # Ensure the blur level does not exceed max_blur
    if blur_level > max_blur:
        blur_level = max_blur


    return blur_level




def plot_loss_and_lr(d_losses, g_losses, d_lrs, g_lrs, num_epochs, save=True, save_dir="results/", show=False, start_epoch=10, interval_epoch=10):
    fig, ax1 = plt.subplots()

    # Losses
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color) 
    if len(d_losses) > 0:
        ax1.plot(range(start_epoch, (len(d_losses) + 1) * interval_epoch, interval_epoch), d_losses, label="Discriminator Loss", color='red')
    ax1.plot(range(start_epoch, (len(g_losses) + 1) * interval_epoch, interval_epoch), g_losses, label="Generator Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the learning rates
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(range(start_epoch, (len(d_lrs) + 1) * interval_epoch, interval_epoch), d_lrs, label="Discriminator LR", color='purple', linestyle='--')
    ax2.plot(range(start_epoch, (len(g_lrs) + 1) * interval_epoch, interval_epoch), g_lrs, label="Generator LR", color='green', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped

    # Save or show the figure
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = os.path.join(save_dir, f"Training_Loss_LR_values_epoch_{num_epochs}.png")
        if len(d_losses) == 0:
            save_fn = os.path.join(save_dir, f"Validation_Loss_LR_values_epoch_{num_epochs}.png")
        plt.savefig(save_fn)
        print(f"Saved plot to {save_fn}")

    if show:
        plt.show()
    else:
        plt.close()



# mean = np.array([0.5, 0.5, 0.5])
# std = np.array([0.5, 0.5, 0.5])

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
