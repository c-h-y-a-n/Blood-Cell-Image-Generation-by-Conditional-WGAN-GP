# Conditional WGAN-GP for Blood Cell Image Synthesis

Generates synthetic blood cell images to address class imbalance in hematology datasets using conditional WGAN-GP.

## Motivation
Blood cell classification suffers from severe class imbalance (e.g., rare basophils, eosinophils).  
Synthetic augmentation via GANs can improve downstream classifier performance on rare classes.

## Dataset
- PBC dataset from Hospital Clinic de Barcelona (https://data.mendeley.com/datasets/snkd93bnjr/1)
- Resolution: 128×128  
- 8 classes: neutrophil, eosinophil, basophil, lymphocyte, monocyte, immature granulocyte (IG), erythroblast, and platelet  
- ~17k total images (~2k per class on average, highly imbalanced)

## Model
- Conditional WGAN-GP with spectral normalization on critic  
- Generator: DCGAN-style with transposed conv
- Latent dim: 100
- Class embedding: 128×128 layer for critic, 100-dim vector for generator 
- Optimizer: Adam (lr=1e-4, β₁=0 (Critic)/ 0.5 (Generator), β₂=0.9)  
- Gradient penalty λ=10  
- Trained for 120 epochs
- Final model backed up in [bloodcell_wgan_gp_ckpt.pth](bloodcell_wgan_gp_ckpt.pth)

## Results
- Final mean per-class FID: ~110 (computed with 2k real + 2k generated per class)  

### Per-class FID (epoch 120)
| Class              | FID    |
|--------------------|--------|
| Neutrophil         | 116.1  |
| Eosinophil         | 102.3  |
| Basophil           | 108.4  |
| Lymphocyte         | 133.7  |
| Monocyte           | 103.1  |
| IG                 | 94.5   |
| Erythroblast       | 105.2  |
| Platelet           | 129.0  |
| **Mean**           | **111.5** |

### Qualitative Results
Balanced generated grids (8 images per class, fixed seed):

**Epoch 30** (early, heavy mode collapse)  
![epoch_30](figures/images/generated_cell_step_30.png)

**Epoch 60** (improved diversity, still blurry)  
![epoch_60](figures/images/generated_cell_step_60.png)

**Epoch 90** (best sharpness)  
![epoch_90](figures/images/generated_cell_step_90.png)

**Epoch 120** (final)  
![epoch_120](figures/images/generated_cell_step_120.png)

## Limitations
- Persistent blurriness on granules and boundaries  
- FID variance due to small per-class sample size (~2k)  
- Rare classes still under-represented

## Future Work
- Add residual blocks + perceptual loss  
- Train at 256×256 resolution  
- Explore diffusion models

## How to Run
```bash
pip install -r requirements.txt
python train.py --data_dir="path/to/dataset" [--resume]
```
- `--epochs` to adjust number of epochs to run
- `--batch_size` to adjust batch size
- `--lr` to adjust learning rate
- `--z_dim` to adjust latent dimension for generator
- `--critic_step` to adjust the number of critic step before one generator step
- `--lambda_gp` to adjust the lambda for gradient penalty
- `--embed_size` to adjust the dimension of class embedding for the generator
- `--adam_beta_1_c`, `--adam_beta_1_g`, `--adam_beta_2` to adjust betas for Adam optimizer
- `--checkpoint_path`, `--log_dir`, `--data_dir` to specify path for checkpoint, SummaryWriter log, and Dataset
- `--seed` to adjust the random seed used
