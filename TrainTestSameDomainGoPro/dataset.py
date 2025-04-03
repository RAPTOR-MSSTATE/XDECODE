# Custom dataset
import PIL
from PIL import Image
import torch.utils.data as data
import os
import random

PIL.Image.MAX_IMAGE_PIXELS = 933120000



class DatasetFromFolder(data.Dataset):
    def __init__(self, root, subfolder, transform=None):
        self.transform = transform
        self.data_dir = os.path.join(root, subfolder)
        self.image_paths = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir) if img.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, image