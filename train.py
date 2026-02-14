import argparse
import random
from collections import defaultdict
import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from IPython import display

from dataset import DEVICE, blood_cell_labels
from models import WGAN_GP
from utils import get_dataloader, save_checkpoint, load_checkpoint, generate_labeled_grid



# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Conditional WGAN-GP Training")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--critic_step', type=int, default=7, help='Crititc Step')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Lambda for Gradient Penalty')
    parser.add_argument('--embed_size', type=int, default=100, help='Class embedding size')
    parser.add_argument('--adam_beta_1_c', type=float, default=0, help='Beta 1 for Adam optimizer (critic)')
    parser.add_argument('--adam_beta_1_g', type=float, default=0.5, help='Beta 1 for Adam optimizer (generator)')
    parser.add_argument('--adam_beta_2', type=float, default=0.9, help='Beta 2 for Adam optimizer')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='bloodcell_wgan_gp_ckpt.pth', help='Path to save/load checkpoint')
    parser.add_argument('--log_dir', type=str, default='runs/bloodcell_experiment')
    parser.add_argument('--data_dir', type=str, default='data/PBC_dataset_normal_DIB')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

IMAGE_SIZE = 128
NUM_CLASSES = 8
CHANNELS = 3

args = parse_args()
print(f"Training with: epochs={args.epochs}, batch_size={args.batch_size}, embed_size={args.embed_size}")


# Fix seed for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# One epoch training function
def train(model, dataloader, c_optim, g_optim, train_metrics):
    model.train()

    for metrics in train_metrics.values():
        metrics.reset()


    total_steps = len(dataloader) // model.critic_step
    iter_dataloader = iter(dataloader)
    pbar = tqdm(range(total_steps), desc="Training WGAN-GP", leave = True)

    for steps in pbar:

        #1 Update Discriminator
        for j in range(model.critic_step):
            model.critic.zero_grad()

            # the real images and fake images
            try:
                train_imgs, labels = next(iter_dataloader)
            except StopIteration:
                iter_dataloader = iter(dataloader)
                train_imgs, labels = next(iter_dataloader)

            train_imgs = train_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            latents = torch.randn(size = (train_imgs.shape[0] , args.z_dim, 1, 1)).to(DEVICE)
            rnd_labels = torch.randint(0, NUM_CLASSES, (train_imgs.shape[0],)).to(DEVICE)
            generated_imgs = model.generator(latents, rnd_labels).detach() # detach to avoid backprop to generator

            # calculate EM distance and gradient penalty
            W_dist = model.critic(train_imgs, labels).mean() - model.critic(generated_imgs, rnd_labels).mean()
            gp = model.gradient_penalty(train_imgs, generated_imgs, labels)
            critic_loss = -W_dist + args.lambda_gp * gp

            critic_loss.backward()
            c_optim.step()


        # 2 Update Generator
        model.generator.zero_grad()

        # random labels for generator training
        latents = torch.randn(size = (train_imgs.shape[0] , args.z_dim, 1, 1)).to(DEVICE)
        rnd_labels = torch.randint(0, NUM_CLASSES, (train_imgs.shape[0],)).to(DEVICE)

        # generate new batch of fake images from latents
        generated_imgs_fresh = model.generator(latents, rnd_labels)

        # calculate generator loss
        g_loss = -model.critic(generated_imgs_fresh, rnd_labels).mean()
        g_loss.backward()
        g_optim.step()


        # 3 Update Metrics
        train_metrics['c_wass_loss'].update(-W_dist.detach().cpu())
        train_metrics['c_gp'].update(gp.detach().cpu())
        train_metrics['g_loss'].update(g_loss.detach().cpu())
        train_metrics['c_loss'].update(critic_loss.detach().cpu())

        pbar.set_postfix({
        "C_Loss": f"{critic_loss:.3f}",
        "G_Loss": f"{g_loss.item():.3f}",
        "GP": f"{gp.item():.4f}"
        })

    return rnd_labels, generated_imgs_fresh.detach().cpu()


# Define the model, dataloader, optimizers, and metrics
wgan_gp = WGAN_GP(args.z_dim, NUM_CLASSES, args.embed_size, CHANNELS, args.critic_step, image_size=IMAGE_SIZE).to(DEVICE)
c_optim = torch.optim.Adam(params=wgan_gp.critic.parameters(), lr=args.lr, betas=(args.adam_beta_1_c, args.adam_beta_2))
g_optim = torch.optim.Adam(params=wgan_gp.generator.parameters() ,lr=args.lr, betas=(args.adam_beta_1_g, args.adam_beta_2))


train_metrics = {
    'c_wass_loss': MeanMetric(),
    'c_gp': MeanMetric(),
    'c_loss': MeanMetric(),
    'g_loss': MeanMetric()
}

bloodcell_dataloader = get_dataloader(args.data_dir, args.batch_size, IMAGE_SIZE, suppress_print = False)
fid_metric = FID(normalize = True).to(DEVICE)
fid_history = list()
history = defaultdict(list)


# for model checkpoint
if args.resume:
    start_epoch, history = load_checkpoint(args.checkpoint_path, wgan_gp.generator, wgan_gp.critic, g_optim, c_optim)
else:
    start_epoch = 0
    print("Starting training from scratch.")

# for logger
writer = SummaryWriter(args.log_dir)


# Training Loop
for i in range(start_epoch, args.epochs):
    prev_time = time.time()
    labels, generated_imgs = train(wgan_gp, bloodcell_dataloader, c_optim, g_optim, train_metrics)
    curr_time = time.time()



    for key, value in train_metrics.items():
        history[key].append(value.compute().item())

    display.clear_output(wait=True)

    print('Epoch: {}\tepoch time {:.2f} min'.format(i+1, (curr_time - prev_time) / 60))
    metrics = [f'{key}: {value.compute().item():.4f} | ' for key, value in train_metrics.items()]
    print('\t', ''.join(metrics))


    # generated_imgs = generated_imgs.detach().cpu()
    # display_imgs(generated_imgs, labels)  # denormalize to [0, 1] for display
    # show_records(history)


    writer.add_scalar('Wasserstein Loss/Critic', train_metrics['c_wass_loss'].compute(), i+1)
    writer.add_scalar('Gradient Penalty/Critic', train_metrics['c_gp'].compute(), i+1)
    writer.add_scalar('Loss/Generator', train_metrics['g_loss'].compute(), i+1)
    writer.add_scalar('Loss/Critic', train_metrics['c_loss'].compute(), i+1)


    if (i + 1) % 10 == 0:
        # Use make_grid to see a batch of LEGOs at once
        labeled_grid = generate_labeled_grid(wgan_gp, 8, NUM_CLASSES, args.z_dim)
        writer.add_image('Generated_Cells', labeled_grid, i+1)

        # Evaluate and record FID
        print(f"\n=== Running FID evaluation at epoch {i + 1}")

        # Full dataset by class in defaultdict
        fid_dataloader = get_dataloader(args.data_dir, args.batch_size, IMAGE_SIZE, suppress_print = True)
        real_per_class = defaultdict(list)
        for batch, (imgs, labels) in enumerate(fid_dataloader):
            for j in range(labels.shape[0]):
                lbl_scalar = labels[j].item()
                global_idx = batch * imgs.shape[0] + j # only the last batch may deviate from original batch size
                real_per_class[lbl_scalar].append(global_idx)

        fid_per_class = {}
        num_samples_per_class = 2000


        for cls in range(NUM_CLASSES):
            fid_metric.reset()

            indices = real_per_class[cls]
            n_real = min(2000, len(indices))
            if n_real == 0:
                continue

            selected_indices = random.sample(indices, n_real) if n_real < len(indices) else indices

            batch_size_gpu = 100  # safe GPU batch size; lower if still OOM

            for start in range(0, n_real, batch_size_gpu):
                end = min(start + batch_size_gpu, n_real)
                batch_idx = selected_indices[start:end]

                # load small batch on CPU
                batch_imgs = [fid_dataloader.dataset[idx][0] for idx in batch_idx]
                batch_imgs = torch.stack(batch_imgs)
                batch_imgs = torch.clamp((batch_imgs + 1)/2, 0.0, 1.0).to(DEVICE)

                # generate matching fakes
                latents = torch.randn(len(batch_idx), args.z_dim, 1, 1, device=DEVICE)
                labels = torch.full((len(batch_idx),), cls, device=DEVICE, dtype=torch.long)
                with torch.no_grad():
                    batch_fakes = wgan_gp.generator(latents, labels)
                batch_fakes = torch.clamp((batch_fakes + 1)/2, 0.0, 1.0)

                # update FID immediately (small chunk)
                fid_metric.update(batch_imgs, real=True)
                fid_metric.update(batch_fakes, real=False)

                # free memory right away
                del batch_imgs, batch_fakes, latents, labels
                torch.cuda.empty_cache()

            fid_per_class[cls] = fid_metric.compute().item()


        mean_fid = sum(fid_per_class.values()) / len(fid_per_class)
        fid_history.append((i + 1, mean_fid))
        print("Per-class FID:", fid_per_class)
        print(f"Mean per-class FID: {mean_fid:.3f}")
        fid_metric.reset()

        # save parameters and metrics to checkpoint
        save_checkpoint(i + 1, wgan_gp.generator, wgan_gp.critic, g_optim, c_optim, history, fid_history, args.checkpoint_path)

        # Add FID and sample images to Summary
        for cls, val in fid_per_class.items():
            writer.add_scalar(f'FID/class_{cls}: {blood_cell_labels[cls]}', val, i+1)
        writer.add_scalar('FID/mean_per_class', mean_fid, i+1)



writer.close()
