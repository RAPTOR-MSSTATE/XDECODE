import argparse
import os
import random
import logging


import torch
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import vgg16
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import albumentations as A
from torchsummary import summary
import numpy as np


import utils
from logger import Logger
from model import Discriminator, Generator, add_spectral_norm

# Configure logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
fh = logging.FileHandler('training_log.log')
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)

# only write what it is in print statement
# Redirect print to logging
def print(*args, **kwargs):
    logger.info(' '.join(map(str, args)))

# ==================================
# Training with no curriculum learning: gradually increasing blur level
# Using Original GoPro blur dataset sharp and inject blur
# ====================================

class ImageFolderNoClass(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.blur_image_paths = []
        self.sharp_image_paths = []
        
        # Process each subdirectory separately
        for subdir, _, _ in os.walk(root):
            # in GOPRO_Large_all it has only sharp images; no folder of sharp images
            sharp_dir = os.path.join(subdir, "sharp")
            if os.path.exists(sharp_dir):
                sharp_images = [os.path.join(sharp_dir, img) for img in os.listdir(sharp_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
                self.sharp_image_paths.extend(sharp_images)

        print("====")
        print(len(self.sharp_image_paths))
        print(self.sharp_image_paths[-5:])
        print("====")

        # # Test out
        # self.sharp_image_paths = self.sharp_image_paths[:200]

    def __len__(self):
        return len(self.sharp_image_paths)

    def __getitem__(self, index):
        sharp_image_path = self.sharp_image_paths[index]
        sharp_image = Image.open(sharp_image_path).convert('RGB')
        if self.transform is not None:
            sharp_image = self.transform(sharp_image)

        return sharp_image, sharp_image




parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='path_to_dataset/', help='path to dataset directory')
parser.add_argument('--batch_size', type=int, default=200, help='train batch size')
parser.add_argument('--ngf', type=int, default=96)
parser.add_argument('--ndf', type=int, default=96)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=256, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=0, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=False, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=100, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.001, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.001, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=2, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
save_dir = 'gopro_results/'
model_dir = 'gopro_model/'


if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

# Data pre-processing

train_transform = transforms.Compose([
    transforms.Resize((params.resize_scale, params.resize_scale)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

valid_transform = transforms.Compose([
    transforms.Resize((params.resize_scale, params.resize_scale)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# Loading datasets
train_data = ImageFolderNoClass(root=os.path.join(params.dataset_dir, 'train'), transform=train_transform)
train_data_loader = DataLoader(dataset=train_data, batch_size=params.batch_size, shuffle=True)

# Create a subset for the first 200 samples
num_valid_samples = 200
valid_data = ImageFolderNoClass(root=os.path.join(params.dataset_dir, 'test'), transform=valid_transform)
valid_data_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)


# Models
G = Generator(3, params.ngf, 3)
D = Discriminator(6, params.ndf, 1)  # 3 for generated and 3 channels for real
G.cuda()
D.cuda()

G.normal_weight_init(mean=0.0, std=0.02)
D.normal_weight_init(mean=0.0, std=0.02)

# add_spectral_norm(D)


print("Generator Summary:")
print(summary(G, (3, 256, 256)))

print("\n\n")


# Set the logger
D_log_dir = save_dir + "D_logs"
G_log_dir = save_dir + "G_logs"
if not os.path.exists(D_log_dir):
    os.makedirs(D_log_dir)
D_logger = Logger(D_log_dir)

if not os.path.exists(G_log_dir):
    os.makedirs(G_log_dir)
G_logger = Logger(G_log_dir)

# Loss functions
L1_loss = torch.nn.L1Loss().cuda()

# Load VGG16 for perceptual loss
vgg = vgg16(pretrained=True).features[:16].cuda()
for param in vgg.parameters():
    param.requires_grad = False

def denormalize_from_first(img):
    # Assume img is in [-1, 1] , normalize with mean/std = [0.5, 0.5, 0.5]
    return (img + 1) * 0.5  # This will bring images to [0, 1]

def normalize_for_vgg(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    return (img - mean) / std  # Normalize to format expected by VGG


def perceptual_loss(fake_img, real_img):
    # Step 1: Denormalize from first normalization scheme ([0.5, 0.5, 0.5]); generator output => [-1, 1]
    fake_img_denorm = denormalize_from_first(fake_img)
    real_img_denorm = denormalize_from_first(real_img)

    # # Step 2: Re-normalize for VGG
    fake_img_vgg = normalize_for_vgg(fake_img_denorm)
    real_img_vgg = normalize_for_vgg(real_img_denorm)
    
    # Extract features using VGG
    fake_features = vgg(fake_img_vgg)
    real_features = vgg(real_img_vgg)
    
    return L1_loss(fake_features, real_features)



# Hinge loss function
def hinge_d_loss(dis_real, dis_fake):
    real_loss = torch.mean(F.relu(1.0 - dis_real))
    fake_loss = torch.mean(F.relu(1.0 + dis_fake))
    return (real_loss + fake_loss)

def hinge_g_loss(dis_fake):
    return -torch.mean(dis_fake)



# Optimizers
G_optimizer = torch.optim.Adam(
    G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2)
)
D_optimizer = torch.optim.Adam(
    D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2)
)

# Learning rate scheduler
G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, mode='min', factor=0.1, patience=25, verbose=True, min_lr=1e-4)
D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, mode='min', factor=0.1, patience=25, verbose=True, min_lr=1e-4)

print("Initial Learning rates: ")
print(f"Generator Learning rate: {D_optimizer.param_groups[0]['lr']}")
print(f"Discriminator Learning rate: {G_optimizer.param_groups[0]['lr']}")

# Blur function
def apply_blur(images, blur_level):
    # Define the blur transformation
    # blur_limit = [0, given_limit)
    # this setting is choosen for curriculum learning
    blur_transform = A.Blur(blur_limit=blur_level, p=0.85) # for the case of curriculum learning

    def __blur(image):
        # Convert PyTorch tensor to numpy array in HWC format for Albumentations
        np_image = image.permute(1, 2, 0).numpy()
        
        # Apply the blur transformation
        augmented = blur_transform(image=np_image)
        
        # Retrieve the blurred image and convert back to PyTorch tensor in CHW format
        blurred_image = augmented['image']
        return torch.from_numpy(blurred_image).permute(2, 0, 1)

    # Apply the blur to each image in the batch
    blurred_images = [__blur(image) for image in images]
    return torch.stack(blurred_images)


# Set the weights for each loss component
hinge_loss_weight = 1.0
l1_loss_weight = 30.0
perc_loss_weight = 1.0

# Training GAN
D_avg_losses = []
G_avg_losses = []
G_val_avg_losses = []
# Lists to store these values
d_lr_list = []
g_lr_list = []
min_G_loss = float("inf")
blur_level = 3
blur_step = 2
blur_epoch_interval = 7
best_val_loss = float("inf")  # Initialize best validation loss to infinity
loss_not_improved_since = 0
model_saved_epoch = 0
patience = 40  # Number of epochs to wait for improvement
delta = 0.02 

step = 0
for epoch in range(params.num_epochs):
    D_losses = []
    G_losses = []

    if (epoch + 1) % blur_epoch_interval == 0:
        blur_level = min(blur_level + blur_step, 19)

    
    # Adjust learning rates with warm-up
    # adjust_learning_rate(G_optimizer, epoch, params.lrG)
    # adjust_learning_rate(D_optimizer, epoch, params.lrD)

    G.train()
    D.train()
    for i, (input, target) in enumerate(train_data_loader):
        # Apply blur to input
        input = apply_blur(input, blur_level)

        # Input & target image data
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        D_fake_decision = D(x_, G(x_).detach()).squeeze()
        D_loss = hinge_d_loss(D_real_decision, D_fake_decision)

        # Back propagation for Discriminator
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = hinge_loss_weight * hinge_g_loss(D_fake_decision)

        # L1 loss
        l1_loss = l1_loss_weight * L1_loss(gen_image, y_)

        # Perceptual loss
        perc_loss = perc_loss_weight * perceptual_loss(gen_image, y_)

        # Combined loss
        G_loss = G_fake_loss + l1_loss + perc_loss #+ ssim_l + color_l + bc_loss

        # Backpropagation for generator
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # Log and print losses
        D_losses.append(D_loss.data.item())
        G_losses.append(G_loss.data.item())

        print(
            "Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f"
            % (
                epoch + 1,
                params.num_epochs,
                i + 1,
                len(train_data_loader),
                D_loss.data.item(),
                G_loss.data.item(),
            )
        ) 

        # TensorBoard logging
        D_logger.scalar_summary("losses", D_loss.data.item(), step + 1)
        G_logger.scalar_summary("losses", G_loss.data.item(), step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # Avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)



    # Get current learning rate from the optimizer
    current_d_lr = D_optimizer.param_groups[0]['lr']
    current_g_lr = G_optimizer.param_groups[0]['lr']
        
    # Append current learning rate to list
    d_lr_list.append(current_d_lr)
    g_lr_list.append(current_g_lr)


    # plot in every epoch: To get idea what going on [monitor losses]
    # Plot average losses
    utils.plot_loss_and_lr(
        D_avg_losses, G_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
    )
    # Validation
    G.eval()  # set to evaluation mode
    D.eval()
    val_losses = []
    print("Calculating validation loss for 100 samples...")
    # random_indices = random.sample(range(len(valid_data)), 100)
    print("Calculating validation loss for first 100 samples...")
    for idx in range(100):
        input, target = valid_data[idx]
        input_blurred = apply_blur(input.unsqueeze(0), blur_level).cuda()
        target = target.unsqueeze(0).cuda()  # Ensure target is 4D

        with torch.no_grad():
            gen_image = G(input_blurred)
            D_fake_decision = D(input_blurred, gen_image).squeeze()
            G_fake_loss = hinge_loss_weight * hinge_g_loss(D_fake_decision)
            l1_loss = l1_loss_weight * L1_loss(gen_image, target)
            perc_loss = perc_loss_weight * perceptual_loss(gen_image, target)
            val_loss = G_fake_loss + l1_loss + perc_loss
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Validation loss for epoch {epoch + 1}: {avg_val_loss}")
    # track of validation loss
    G_val_avg_losses.append(avg_val_loss)
    # plot validation loss in every epoch
    utils.plot_loss_and_lr(
        [], G_val_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
    )


    # # Update learning rate with validation loss
    G_scheduler.step(G_avg_loss)
    D_scheduler.step(G_avg_loss)
    
    # removed early stopping criteria

    # Save the model if the validation loss decreases
    if avg_val_loss <= best_val_loss:
        print(f"Validation loss decreased ({best_val_loss} --> {avg_val_loss}). Saving model...")
        best_val_loss = avg_val_loss
        loss_not_improved_since = 0  # Reset counter if there is improvement
        model_saved_epoch = epoch + 1
        torch.save(G.state_dict(), os.path.join(model_dir, 'generator_best_param.pkl'))
        torch.save(D.state_dict(), os.path.join(model_dir, 'discriminator_best_param.pkl'))

    
    else:
        loss_not_improved_since += 1
        print(f"Validation loss increased (Best Loss: {best_val_loss} --> {avg_val_loss}). Skip Saving ...")
        print(f"Patience counter: {loss_not_improved_since}")
        print(f"Generator Learning rate: {D_optimizer.param_groups[0]['lr']}")
        print(f"Discriminator Learning rate: {G_optimizer.param_groups[0]['lr']}")
        torch.save(G.state_dict(), os.path.join(model_dir, 'generator_param.pkl'))
        torch.save(D.state_dict(), os.path.join(model_dir, 'discriminator_param.pkl'))
        # temp_dir = f"{model_dir}/epoch_{epoch+1}" # f"model_dir/{epoch+1}"
        # os.makedirs(temp_dir, exist_ok=True)
        # torch.save(G.state_dict(), os.path.join(temp_dir, 'generator_param.pkl'))
        # torch.save(D.state_dict(), os.path.join(temp_dir, 'discriminator_param.pkl'))
    
    
    # Early stopping
    # if loss_not_improved_since >= patience:
    #     print(f"No improvement since {patience} epochs. Stopping training...")
    #     print(f"Final time model saved is in : {model_saved_epoch}")
    #     break

    # Save validation images
    epoch_save_dir = os.path.join("gopro_validation_results/", f'epoch_{epoch + 1}')
    os.makedirs(epoch_save_dir, exist_ok=True)

    print("\n Saving random 20 images...")
    random_indices = random.sample(range(len(valid_data)), 20)
    for idx in random_indices:
        input, target = valid_data[idx]
        input_blurred = apply_blur(input.unsqueeze(0), blur_level).cuda()
        with torch.no_grad():
            gen_image = G(input_blurred)

        # used for plotting
        gen_image = gen_image.cpu().data

        # Calculate SSIM and PSNR
        target_np = target.numpy().transpose(1, 2, 0)
        generated_np = gen_image.squeeze().numpy().transpose(1, 2, 0)
        
        ssim_cal = structural_similarity(
            target_np,
            generated_np,
            multichannel=True,
            data_range=target_np.max() - target_np.min(),
            win_size=7,
            channel_axis=2
        )
        psnr_cal = peak_signal_noise_ratio(target_np, generated_np, data_range=target_np.max() - target_np.min())

        # Save the result
        result_image_name = f'{idx}_blur_{blur_level}.png'
        utils.plot_test_result(
            input_blurred.cpu(),
            target.unsqueeze(0).cpu(),
            gen_image,
            idx,
            training=False,
            save=True,
            save_dir=epoch_save_dir,
            show=False,
            SSIM=ssim_cal,
            PSNR=psnr_cal
        )
        print(f'Saved validation image {result_image_name} for epoch {epoch + 1} with blur level {blur_level}')

# Plot average losses
utils.plot_loss_and_lr(
    D_avg_losses, G_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
)

utils.plot_loss_and_lr(
    [], G_val_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
)

print(f"Final time model saved is in : {model_saved_epoch}")
print("Final Learning rates: ")
print(f"Final Generator Learning rate: {G_optimizer.param_groups[0]['lr']}")
print(f"Final Discriminator Learning rate: {D_optimizer.param_groups[0]['lr']}")
# Make gif
# utils.make_gif(params.dataset_dir, params.num_epochs, save_dir=save_dir)
