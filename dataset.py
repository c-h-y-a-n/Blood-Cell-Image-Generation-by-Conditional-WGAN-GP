import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms as Transforms
from PIL import Image, ImageDraw, ImageFont


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

blood_cell_labels = [
    "neutrophil",
    "eosinophil",
    "basophil",
    "lymphocyte",
    "monocyte",
    "ig",
    "erythroblast",
    "platelet"
]


class BloodCellData(Dataset): # from torch.utils.data
    def __init__(self, img_dir, image_size):
        super().__init__()
        self.img_dir = img_dir
        self.image_size = image_size

        self.imgs = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith('.jpeg') or file.endswith('.jpg'):
                    self.imgs.append(os.path.join(root, file))
        
        self.length = len(self.imgs)
        
        self.blood_cell_type = list(map(os.path.basename, map(os.path.dirname, self.imgs)))
        
        self.label = list(map(lambda x: blood_cell_labels.index(x), self.blood_cell_type))

        # Define Transformer
        self.transform = Transforms.Compose([
            Transforms.Lambda(lambda img: img.convert('RGB')),
            Transforms.ToTensor(),
            Transforms.Resize((self.image_size, self.image_size)),
            Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])



    def __len__(self):
        return self.length

    def __getitem__(self, idx): # enable indexing
        img_file = self.imgs[idx]
        image = Image.open(img_file)
        image = self.transform(image)
        label = self.label[idx]
        return image, label