import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import einops
from einops import rearrange, repeat
import yaml

torch.manual_seed(0)


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, patch_size=16, tubelet_size=2, image_size=224):  # TODO: get right hyperparameters
        super(SpatialTransformer, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv_proj = nn.Conv3d(
            in_channels=3, out_channels=embed_dim, kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # TODO: randn vs zeros

        seq_length = (image_size // patch_size) ** 2
        seq_length += 1  # For cls token
        self.pos_embed = nn.Parameter(torch.empty(1, seq_length, embed_dim).normal_(std=0.02))  # from BERT

    def forward(self, x):
        b, t, c, h, w = x.shape
        n = b*t

        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        torch._assert(t % self.tubelet_size == 0, f"Wrong video length! {t} must be dividable by {self.tubelet_size}!")

        n_h = h // self.patch_size
        n_w = w // self.patch_size

        # Reshape input for Conv3d
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.conv_proj(x)

        # Merge batch+time and height+width dimensions
        x = rearrange(x, 'b d t h w -> (b t) (h w) d')

        # Expand the class token to the full batch
        cls_token = repeat(self.cls_token, '() np d -> n np d', n=int(n/self.tubelet_size))
        x = torch.cat([cls_token, x], dim=1)

        # Add positional encoding
        x += self.pos_embed[:, :(n_h*n_w) + 1]

        x = self.encoder(x)

        # Get class token
        x = x[:, 0]

        x = rearrange(x, '(b t) ... -> b t ...', b=b)  # Unflatten batch and temporal dims

        return x  # Return the class token (batch_size, n_frames, embed_dim)


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, seq_length=16):  # TODO: get right hyperparameters
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # TODO: randn vs zeros
        # TODO: what to use for positional embedding
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, 100, embed_dim))  # Adjust the 100 according to your max sequence length
        self.pos_embed = nn.Parameter(torch.empty(1, seq_length + 1, embed_dim).normal_(std=0.02))  # from BERT

    def forward(self, x):
        b, t, n = x.shape

        # Add temporal CLS token
        cls_token = repeat(self.cls_token, '() t n -> b t n', b=b)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional encoding
        x += self.pos_embed[:, :t + 1]

        x = self.encoder(x)

        return x[:, 0]  # Return the class token (B, E)


class ViViT(nn.Module):
    def __init__(self, config):
        super(ViViT, self).__init__()
        self.spatial_transformer = SpatialTransformer(embed_dim=config['embed_dim'],
                                                        num_heads=config['spatial_num_heads'],
                                                        num_layers=config['spatial_num_layers'],
                                                        patch_size=config['patch_size'],
                                                        tubelet_size=config['tubelet_size'],
                                                        image_size=config['image_size'])
        self.temporal_transformer = TemporalTransformer(embed_dim=config['embed_dim'],
                                                        num_heads=config['temporal_num_heads'],
                                                        num_layers=config['temporal_num_layers'],
                                                        seq_length=config['max_seq_length'])
        # self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config['embed_dim']),
            nn.Linear(config['embed_dim'], config['num_classes'])
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        # Process each frame with Spatial Transformer
        spatial_embeddings = self.spatial_transformer(x)

        # Process frame embeddings with Temporal Transformer
        video_embedding = self.temporal_transformer(spatial_embeddings)

        # Classification
        logits = self.classifier(video_embedding)
        return logits


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


if __name__ == '__main__':

    config_path = 'config.yaml'
    config = load_config(config_path)

    data_config = config['data']

    # Example usage
    video_data = torch.randn(data_config['batch_size'], data_config['num_frames'], data_config['num_channels'],
                             data_config['height'], data_config['width']).cuda()
    model = ViViT(config['model']).cuda()

    logits = model(video_data)
    print(logits.shape)  # Output: torch.Size([batch_size, num_classes])
