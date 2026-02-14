import torch
from torch import nn
from torch.nn.utils import spectral_norm


from dataset import DEVICE

# Critic Network
class Critic(nn.Module):
    def __init__(self, num_classes, channels, image_size):
        super().__init__()

        channel_list = [channels + 1, 64, 128, 256, 512, 1024] # additional channel for labels

        self.conv_layers = nn.Sequential()
        for i in range(len(channel_list) - 1):
            conv_block = self.get_conv_block(channel_list[i], channel_list[i+1], bool(i))
            self.conv_layers.add_module(name = f'conv_block_{i+1}', module = conv_block)

        self.output_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels = channel_list[-1], out_channels = 1, kernel_size=2, stride=1, padding=0, bias=False)),
            nn.Flatten()
        )

        self.image_size = image_size

        self.embed = nn.Embedding(num_classes, self.image_size*self.image_size)


    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1) # concatenate in the CHANNEL dimension (batch size, CHANNEL, H, W)
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x
    def get_conv_block(self, in_channels, out_channels, use_in = True):
        layers = [
            spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace = False)
            # nn.Dropout2d(p = 0.3)
        ]
        if use_in:
            layers.insert(1, nn.InstanceNorm2d(out_channels))
        return nn.Sequential(*layers) # the * to unpack the list argument
    

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, embed_size, channels):
        super().__init__()

        channel_list = [z_dim + embed_size, 1024, 512, 256, 128, 64] # extra channels for labels

        self.conv_trans_layers = nn.Sequential()
        for i in range(len(channel_list) - 1):
            stride = 2 if i else 1
            padding = 1 if i else 0
            trans_conv_block = self.get_conv_trans_block(channel_list[i], channel_list[i+1], stride, padding)
            self.conv_trans_layers.add_module(name = f'conv_trans_block_{i+1}',
                                              module = trans_conv_block)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels = channel_list[-1], out_channels = channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.z_dim = z_dim
        self.embed_size = embed_size

        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, x, labels):
        x = x.view(-1, self.z_dim, 1, 1)
        # reshape to (batch size, noise_dim, 1, 1)
        embedding = self.embed(labels).view(-1, self.embed_size, 1, 1)
        x = torch.cat([x, embedding], dim=1)
        x = self.conv_trans_layers(x)
        x = self.output_layer(x)
        return x

    def get_conv_trans_block(self, in_channels, out_channels, stride = 2, padding = 1):
        Layers = [
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size=4, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features = out_channels, affine = True),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*Layers)


class WGAN_GP(nn.Module):
    def __init__(self, z_dim, num_classes, embed_size, channels, critic_steps, image_size):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim, num_classes, embed_size, channels)
        self.critic = Critic(num_classes, channels, image_size)
        self.critic_step = critic_steps


    # def generate(self, real_imgs):
    #     latents = torch.randn(size = (len(real_imgs), self.z_dim, 1, 1)).to(DEVICE)
    #     generated_imgs = self.generator(latents)
    #     return generated_imgs

    def gradient_penalty(self, real_imgs, fake_imgs, labels):
        batch_size = real_imgs.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(DEVICE) # random weight for interpolation
        interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
        interpolated.requires_grad_(True)

        critic_interpolated = self.critic(interpolated, labels)

        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated).to(DEVICE),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty