import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import csv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import utils
import argparse
from model import Generator

class ImageFolderNoClass(data.Dataset):
    def __init__(self, blur_root, sharp_root, transform=None):
        self.blur_root = blur_root
        self.sharp_root = sharp_root
        self.transform = transform
        self.blur_image_paths = []
        self.sharp_image_paths = []

        # Process both blur and sharp images
        self.blur_image_paths = [os.path.join(self.blur_root, img) for img in os.listdir(self.blur_root) if img.endswith(('png', 'jpg', 'jpeg'))]
        self.sharp_image_paths = [os.path.join(self.sharp_root, img) for img in os.listdir(self.sharp_root) if img.endswith(('png', 'jpg', 'jpeg'))]

        # Ensure that both directories have the same images
        assert len(self.blur_image_paths) == len(self.sharp_image_paths), "Mismatch between blurred and sharp images"
        self.blur_image_paths.sort()
        self.sharp_image_paths.sort()

    def __len__(self):
        return len(self.blur_image_paths)

    def __getitem__(self, index):
        blur_image_path = self.blur_image_paths[index]
        sharp_image_path = self.sharp_image_paths[index]

        blur_image = Image.open(blur_image_path).convert('RGB')
        sharp_image = Image.open(sharp_image_path).convert('RGB')

        if self.transform is not None:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image, str(self.blur_image_paths[index])

def save_image(tensor, save_path):
    """Convert a tensor to an image and save it using PIL."""
    image = tensor.squeeze(0).permute(1, 2, 0).numpy()  # Convert CHW to HWC format
    image = ((image * 0.5) + 0.5) * 255.0  # De-normalize from [-1, 1] to [0, 255]
    image = image.astype(np.uint8)
    pil_image = Image.fromarray(image)
    pil_image.save(save_path)

def process_blur_level(blur_level, blur_dir, sharp_dir, save_dir, generator, transform):
    # Create Dataset and limit to first 500 samples
    dataset = ImageFolderNoClass(blur_root=blur_dir, sharp_root=sharp_dir, transform=transform)
    dataset = Subset(dataset, list(range(min(500, len(dataset)))))  # Take the first 500 images

    # Create DataLoader for the current blur level
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)


    # Create directories to save individual images and plots
    deblurred_images_dir = os.path.join(save_dir, "deblurred_images")
    plot_results_dir = os.path.join(save_dir, "plot_results")
    os.makedirs(deblurred_images_dir, exist_ok=True)
    os.makedirs(plot_results_dir, exist_ok=True)

    # Lists to store results
    psnr_calcs = []
    ssim_calcs = []
    result_image_names = []

    for i, (input, target, blur_image_path) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Generate image using the generator
        with torch.no_grad():
            generated_image = generator(input.cuda())

        # Convert to CPU tensors for further processing
        gen_image = generated_image.cpu().data

        # Calculate SSIM and PSNR
        target_np = target.squeeze(0).numpy().transpose(1, 2, 0)
        generated_np = gen_image.squeeze(0).numpy().transpose(1, 2, 0)

        ssim_cal = structural_similarity(
            target_np,
            generated_np,
            multichannel=True,
            data_range=target_np.max() - target_np.min(),
            win_size=7,
            channel_axis=2
        )
        psnr_cal = peak_signal_noise_ratio(target_np, generated_np, data_range=target_np.max() - target_np.min())

        # Save individual deblurred image
        deblurred_image_name = blur_image_path[0].split('/')[-1] # tuple arrary to only file name
        # print("Blur Image Path: ")
        # print(blur_image_path)
        # print(type(blur_image_path))
        # print(blur_image_path[0])
        deblurred_image_path = os.path.join(deblurred_images_dir, deblurred_image_name)
        save_image(gen_image, deblurred_image_path)

        # Print SSIM and PSNR values
        print(f"SSIM: {ssim_cal}, PSNR: {psnr_cal}, Blur Level: {blur_level}")

        # Store the results
        ssim_calcs.append(ssim_cal)
        psnr_calcs.append(psnr_cal)
        result_image_names.append(deblurred_image_name)

        # Plot and save the comparison result
        utils.plot_test_result(
            input,
            target,
            gen_image, 
            i,
            training=False,
            save=True,
            save_dir=plot_results_dir,
            show=False,
            SSIM=ssim_cal,
            PSNR=psnr_cal
        )

    # Calculate mean SSIM and PSNR
    mean_ssim = np.mean(ssim_calcs)
    mean_psnr = np.mean(psnr_calcs)

    print(f"Mean SSIM for blur level {blur_level}: {mean_ssim}")
    print(f"Mean PSNR for blur level {blur_level}: {mean_psnr}")

    # Write results to CSV
    csv_filename = os.path.join(save_dir, f'test_results_blur_level_{blur_level}.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'SSIM', 'PSNR', 'Blur Level'])

        # Write individual SSIM and PSNR values
        for i in range(len(ssim_calcs)):
            writer.writerow([result_image_names[i], ssim_calcs[i], psnr_calcs[i], blur_level])

        # Write mean SSIM and PSNR at the end
        writer.writerow(["Mean", mean_ssim, mean_psnr, blur_level])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='path/to/dataset/', help='path to dataset directory')
    parser.add_argument('--model_dir', default='path/to/model/', help='path to model directory')
    parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--ngf', type=int, default=96)
    parser.add_argument('--resize_scale', type=int, default=256, help='input size')  # Adjusted input size to 256
    params = parser.parse_args()

    print("Given params: \n", params)
    save_dir = "kitti_test_results_self_blurred_new_crop_range_blur/"
    model_dir = params.model_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    valid_transform = transforms.Compose([
        transforms.Resize((params.resize_scale, params.resize_scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Initialize model
    generator = Generator(3, params.ngf, 3)
    generator.cuda()
    generator.eval()
    generator.load_state_dict(torch.load(model_dir + 'generator_best_param.pkl'))


    blur_dir = os.path.join(params.dataset_dir, "blurred")
    sharp_dir = os.path.join(params.dataset_dir, 'sharp')
    level_save_dir = os.path.join(save_dir, "blurred_range")
    # Ensure directory exists for each blur level
    os.makedirs(level_save_dir, exist_ok=True)
    blur_level = "all_blur"
    process_blur_level(blur_level, blur_dir, sharp_dir, level_save_dir, generator, valid_transform)

if __name__ == "__main__":
    main()
