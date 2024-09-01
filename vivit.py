import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import einops
from einops import rearrange, repeat
import yaml

torch.manual_seed(0)


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Using a pretrained Vision Transformer model from torchvision
        self.vit = vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        self.vit.heads = nn.Identity()  # Remove the classification head

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = 16  # TODO
        # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        conv_proj = nn.Conv2d(
            in_channels=3, out_channels=768, kernel_size=16, stride=16
        ).cuda()  # TODO

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, 768, n_h * n_w)  # TODO

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')  # flatten batch and temporal dims

        # Pass through the model
        # torchvision ViT will take care of embeddings and cls token
        # x = self.vit(x)

        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        class_token = nn.Parameter(torch.zeros(1, 1, 768)).cuda()  # TODO
        batch_class_token = class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True).cuda()  # TODO
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12).cuda()  # TODO

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = rearrange(x, '(b t) ... -> b t ...', b=b)  # unflatten batch and temporal dims

        return x  # Return the class token (batch_size, n_frames, embed_dim)


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, seq_length=16):  # TODO: get right hyperparameters
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
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
        self.spatial_transformer = SpatialTransformer()
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
    # Example usage

    config_path = 'config.yaml'
    config = load_config(config_path)

    # batch_size = 2
    # num_frames = 8
    # num_channels = 3
    # height = 224
    # width = 224
    # num_classes = 10

    data_config = config['data']

    video_data = torch.randn(data_config['batch_size'], data_config['num_frames'], data_config['num_channels'],
                             data_config['height'], data_config['width']).cuda()
    model = ViViT(config['model']).cuda()

    logits = model(video_data)
    print(logits.shape)  # Output: torch.Size([batch_size, num_classes])
