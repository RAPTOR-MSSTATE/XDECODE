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
from utils import *
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


# Training dataset class
class ImageFolderNoClass(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.sharp_image_paths = []

        # Process images directly from the root folder
        self.sharp_image_paths = [
            os.path.join(root, img) for img in os.listdir(root)
            if img.endswith(('png', 'jpg', 'jpeg'))
        ]

        print("====")
        print(f"Total images found: {len(self.sharp_image_paths)}")
        print("Sample image paths:", self.sharp_image_paths[-5:])
        print("====")


        # test
        # self.sharp_image_paths = self.sharp_image_paths[:10]
        
    def __len__(self):
        return len(self.sharp_image_paths)

    def __getitem__(self, index):
        sharp_image_path = self.sharp_image_paths[index]
        sharp_image = Image.open(sharp_image_path).convert('RGB')
        if self.transform is not None:
            sharp_image = self.transform(sharp_image)

        return sharp_image, sharp_image


# Validation image class
class ImageFolderNoClassValidation(data.Dataset):
    """
    Return sharp, blur image pairs for validation.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.sharp_image_paths = []
        self.blur_image_paths = []

        # Process the sharp and blur subdirectories
        sharp_dir = os.path.join(root, "sharp")
        blur_dir = os.path.join(root, "blur")

        if os.path.exists(sharp_dir) and os.path.exists(blur_dir):
            self.sharp_image_paths = sorted(
                [os.path.join(sharp_dir, img) for img in os.listdir(sharp_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
            )
            self.blur_image_paths = sorted(
                [os.path.join(blur_dir, img) for img in os.listdir(blur_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
            )

        # Ensure both directories contain the same number of images
        if len(self.sharp_image_paths) != len(self.blur_image_paths):
            raise ValueError("Mismatch between sharp and blur image counts.")

        print(f"Sharp images: {len(self.sharp_image_paths)}")
        print(f"Blur images: {len(self.blur_image_paths)}")

    def __len__(self):
        return len(self.sharp_image_paths)

    def __getitem__(self, index):
        sharp_image_path = self.sharp_image_paths[index]
        blur_image_path = self.blur_image_paths[index]

        sharp_image = Image.open(sharp_image_path).convert('RGB')
        blur_image = Image.open(blur_image_path).convert('RGB')

        if self.transform is not None:
            sharp_image = self.transform(sharp_image)
            blur_image = self.transform(blur_image)

        return blur_image, sharp_image




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='path/to/dataset/', help='path to dataset directory')
    parser.add_argument('--curr_learning', default='linear', help='Which curriculum Learning to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='train batch size')
    parser.add_argument('--ngf', type=int, default=96)
    parser.add_argument('--ndf', type=int, default=96)
    parser.add_argument('--input_size', type=int, default=256, help='input size')
    parser.add_argument('--resize_scale', type=int, default=256, help='resize scale (0 is false)')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of train epochs')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
    parser.add_argument('--lamb', type=float, default=2, help='lambda for L1 loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='To use scheduler, set flag. Change initial lr to 0.001 if using scheduler.')
    parser.add_argument('--save_model_in_epochs', type=int, default=10, help="Save model every N epochs.")
    return parser.parse_args()

params = get_arguments()
print(params)
# Directories for loading data and saving results
save_dir = 'results/'
model_dir = 'model/'

# Supported curriculum learning methods
supported_methods = ["linear", "expo", "sigmoid", "stepwise", "slow_stepwise", "none"]
if params.curr_learning not in supported_methods:
    raise ValueError(f"Invalid curriculum learning method: {params.curr_learning}. Supported: {supported_methods}")

training_method = params.curr_learning

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
valid_data = ImageFolderNoClass(root=os.path.join(params.dataset_dir, 'valid'), transform=valid_transform)
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

if params.use_scheduler:
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

    if training_method == "none":
        # train without curriculum leanring 
        blur_transform = A.Blur(blur_limit=(blur_level,blur_level), p=0.85)
    

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


# Curriculum learning update function
def update_blur_level(curr_blur_level, curr_epoch):
    """Update blur level based on the curriculum learning method."""
    if training_method == "linear":
         # linear curriculum learning
        return min(blur_level + 2, 29) # reached to max blur 29 at epoch 13
    
    elif training_method == "expo":
        return controlled_exponential_curriculum_odd(curr_epoch)
    

    elif training_method == "stepwise":
        if (curr_epoch + 1) % blur_epoch_interval == 0:
            return min(curr_blur_level + blur_step, 29)  # Increase every `blur_epoch_interval`
        return curr_blur_level
    

    elif training_method == "slow_stepwise":
        if (curr_epoch + 1) % (blur_epoch_interval * 2) == 0:
            return min(curr_blur_level + blur_step, 29)  # Slower increase every `2 * blur_epoch_interval`
        return curr_blur_level
    
    elif training_method == "sigmoid":
        return sigmoid_curriculum_odd(curr_epoch=curr_epoch, total_epochs=params.num_epochs)
    

    return 29  # No curriculum learning

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
blur_level = 3 # Initial blur level for curriculum learning
blur_step = 2
blur_epoch_interval = 5  # Stepwise increase every N epochs
save_model_in_every = params.save_model_in_epochs # save model in every 10th epoch.

step = 0
for epoch in range(params.num_epochs):
    D_losses = []
    G_losses = []

    if training_method == "none": # without curriculum method
        blur_level = 29
    else: # with curriculum learning
        blur_level = update_blur_level(blur_level, epoch)

    print(f"Epoch {epoch+1}: Blur Level = {blur_level}:\n")

    G.train()
    D.train()
    for i, (input, target) in enumerate(train_data_loader):
        # Apply blur to input
        input = apply_blur(input, blur_level)

        # Input & target image data
        # Set non_blocking=True when transferring data to GPU to overlap CPU-GPU operations.
        x_ = Variable(input.cuda(non_blocking=True))
        y_ = Variable(target.cuda(non_blocking=True))

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
        D_logger.scalar_summary("losses", D_loss.item(), step + 1)
        G_logger.scalar_summary("losses", G_loss.item(), step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    if (epoch + 1) % save_model_in_every == 0:
        print(f"Saving loss in {epoch + 1}")
        

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

    # validation losses
    val_losses = []
    print("Calculating validation loss...")
    num_save_validation_images = 30 # number of validation images to save

    # Save validation images directory
    epoch_save_dir = os.path.join("cityscapes_validation_results/", f'epoch_{epoch + 1}')
    

    for idx in range(len(valid_data)):
        input_blurred, target = valid_data[idx]
        input_blurred = input_blurred.unsqueeze(0).cuda()
        target = target.unsqueeze(0).cuda()  # Ensure target is 4D
        # print(input_blurred.size(), target.size())

        with torch.no_grad():
            gen_image = G(input_blurred)
            D_fake_decision = D(input_blurred, gen_image).squeeze()
            G_fake_loss = hinge_loss_weight * hinge_g_loss(D_fake_decision)
            l1_loss = l1_loss_weight * L1_loss(gen_image, target)
            perc_loss = perc_loss_weight * perceptual_loss(gen_image, target)
            val_loss = G_fake_loss + l1_loss + perc_loss
            val_losses.append(val_loss.item())

        if num_save_validation_images!=0 and (epoch + 1) % save_model_in_every == 0: # save the generated validation images in every some epochs
            
            if num_save_validation_images == 0: # create directory at first time
                os.makedirs(epoch_save_dir, exist_ok=True)

            num_save_validation_images -= 1 # decrease counter

            # used data for plotting
            gen_image = gen_image.cpu().data

            # Calculate SSIM and PSNR
            target_np = target.cpu().squeeze().numpy().transpose(1, 2, 0)
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
                target.cpu(),
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

    
    # saved validation loss in every ith epoch
    if (epoch + 1) % save_model_in_every == 0:
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation loss for epoch {epoch + 1}: {avg_val_loss}")


        # track of validation loss
        G_val_avg_losses.append(avg_val_loss)
        # plot validation loss in every epoch
        utils.plot_loss_and_lr(
            [], G_val_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
        )


    # # Update learning rate with validation loss
    if params.use_scheduler:
        G_scheduler.step(G_avg_loss)
        D_scheduler.step(G_avg_loss)

    
    if (epoch + 1) % save_model_in_every == 0:  
        print(f"Saving model in {epoch}")
        temp_dir = f"{model_dir}/epoch_{epoch+1}" 
        os.makedirs(temp_dir)
        torch.save(G.state_dict(), os.path.join(temp_dir, 'generator_best_param.pkl'))
        torch.save(D.state_dict(), os.path.join(temp_dir, 'discriminator_best_param.pkl'))


# Plot final average losses
utils.plot_loss_and_lr(
    D_avg_losses, G_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
)

utils.plot_loss_and_lr(
    [], G_val_avg_losses, d_lr_list, g_lr_list, params.num_epochs, save=True, save_dir=save_dir
)

print("Final Learning rates: ")
print(f"Final Generator Learning rate: {G_optimizer.param_groups[0]['lr']}")
print(f"Final Discriminator Learning rate: {D_optimizer.param_groups[0]['lr']}")

# Make gif
# utils.make_gif(params.dataset_dir, params.num_epochs, save_dir=save_dir)
