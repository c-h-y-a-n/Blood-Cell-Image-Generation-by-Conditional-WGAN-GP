import torch
import os
from collections import defaultdict
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as Transforms
import torchvision
from matplotlib import pyplot as plt


from dataset import DEVICE, blood_cell_labels, BloodCellData





def save_checkpoint(epoch, generator, critic, g_optim, c_optim, history, fid_history, path="/content/drive/MyDrive/dcgan_checkpoint.pth"):
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'g_optimizer_state_dict': g_optim.state_dict(),
        'c_optimizer_state_dict': c_optim.state_dict(),
        'history': history,
        'fid_history': fid_history
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"--- Checkpoint saved at epoch {epoch} to {path} ---")

def load_checkpoint(path, generator, critic, g_optim, c_optim):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location = DEVICE)

        generator.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        g_optim.load_state_dict(checkpoint['g_optimizer_state_dict'])
        c_optim.load_state_dict(checkpoint['c_optimizer_state_dict'])

        generator.to(DEVICE)
        critic.to(DEVICE)

        start_epoch = checkpoint.get('epoch', 0) + 1
        history = checkpoint.get('history', defaultdict(list))
        fid_history = checkpoint.get('fid_history', [])

        print(f"Resuming from epoch {start_epoch}")
        return start_epoch, history
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, defaultdict(list)


def get_dataloader(data_dir, batch_size, image_size, suppress_print = False):
    blood_cell_dataset = BloodCellData(data_dir, image_size)
    train_loader = DataLoader(blood_cell_dataset, batch_size=batch_size, shuffle=True, num_workers = 0, pin_memory = True)
    # pin_memory when we read data to CPU, it will be faster to transfer to GPU
    # num_workers defines the number of parallel subprocesses used to load data (optimal: 2 * number of GPU cores, but not recommedned for Windows)
    if not suppress_print:
        print('Train data size: ', len(blood_cell_dataset))
        print('Num. train batchs: ', len(train_loader))
    return train_loader

def generate_labeled_grid(model, image_per_class, num_classes, z_dim):


    # Create balanced labels
    balanced_labels = torch.cat([
        torch.full((image_per_class,), cls, dtype=torch.long, device=DEVICE)
        for cls in range(num_classes)
    ])


    latents = torch.randn(image_per_class * num_classes, z_dim, 1, 1, device=DEVICE)

    with torch.no_grad():
        generated_imgs = model.generator(latents, balanced_labels)
        generated_imgs = torch.clamp((generated_imgs + 1)/2, 0.0, 1.0).detach().cpu()

    # Convert to PIL images
    to_pil = Transforms.ToPILImage()
    pil_imgs = [to_pil(img) for img in generated_imgs]


    # Adjust Font
    font = ImageFont.load_default(size = 15)

    # Add label to each image
    for pil_img, lbl in zip(pil_imgs, balanced_labels):
        draw = ImageDraw.Draw(pil_img)
        text = blood_cell_labels[lbl.item()]
        draw.text((10, 10), text, fill="white", font=font, stroke_width=2, stroke_fill="black")

    # Convert back to tensor grid
    tensor_imgs = [Transforms.ToTensor()(pil_img) for pil_img in pil_imgs]
    labeled_grid = torchvision.utils.make_grid(tensor_imgs, nrow=image_per_class, normalize=True)

    return labeled_grid