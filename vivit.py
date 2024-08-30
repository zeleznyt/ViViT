import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import einops
from einops import rearrange, repeat

torch.manual_seed(0)

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Using a pretrained Vision Transformer model from torchvision
        self.vit = vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        self.vit.heads = nn.Identity()  # Remove the classification head

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w') # flatten batch and temporal dims

        # Pass through the model
        # torchvision ViT will take care of embeddings and cls token
        x = self.vit(x)
        x = rearrange(x, '(b t) ... -> b t ...', b=b) # unflatten batch and temporal dims

        return x # Return the class token (batch_size, n_frames, embed_dim)


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=12, seq_length=16): # TODO: get right hyperparameters
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

        return x[:,0]  # Return the class token (B, E)


class ViViT(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1000, max_seq_len=16):
        super(ViViT, self).__init__()
        self.spatial_transformer = SpatialTransformer()
        self.temporal_transformer = TemporalTransformer(embed_dim=embed_dim, num_heads=8, num_layers=6,
                                                        seq_length=max_seq_len)
        # self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
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


# Example usage
batch_size = 2
num_frames = 8
num_channels = 3
height = 224
width = 224
num_classes = 10

video_data = torch.randn(batch_size, num_frames, num_channels, height, width).cuda()
model = ViViT(num_classes=num_classes).cuda()

logits = model(video_data)
print(logits.shape)  # Output: torch.Size([batch_size, num_classes])
