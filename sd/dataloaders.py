import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from PIL import Image

def read_ppm(filename):
    image = Image.open(filename)
    return np.array(image)

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(read_ppm(f))
            labels.append(int(d))
    return images, labels

class ReshapeTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return img.permute(1, 2, 0).contiguous() 

def transform_image(image, size=(128, 128)):
    return transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    transforms.ToTensor(),
    # ReshapeTransform((512, 512, 3))
])


class TrafficSignsDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=True, size=(128, 128)):
        self.images = images
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        transform = transform_image(image, self.size)
        if self.transform:
            image = transform(image)
        return image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data()
        
    def load_data(self):
        # Implement loading of your dataset
        # This should return a list of data items (e.g., image paths)
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load the image and apply transformations
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image