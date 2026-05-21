import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import vit_b_16

from utils.train_utils import load_config
from vivit import ViViT


CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


class TemporalOnnxWrapper(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model = ViViT(model_config, use_only_embeddings=True)

    def forward(self, embeddings, padding_mask):
        return self.model(embeddings, padding_mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Export ViViT predictor checkpoints to ONNX.")
    parser.add_argument("--config", required=True, help="Path to the training/evaluation config YAML.")
    parser.add_argument("--checkpoint", help="Path to the .pth checkpoint. Defaults to config['evaluation']['checkpoint'].")
    parser.add_argument("--output-dir", help="Directory where the ONNX bundle will be written.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def build_models(model_config, checkpoint_path, device):
    if model_config.get("use_pretrained_encoder") != "vit":
        raise ValueError(
            "This exporter currently supports predictor-compatible checkpoints with "
            "model.use_pretrained_encoder: vit."
        )

    spatial_encoder = vit_b_16(weights=None)
    spatial_encoder.heads = nn.Identity()
    spatial_encoder = spatial_encoder.to(device).eval()

    temporal_encoder = TemporalOnnxWrapper(model_config).to(device).eval()

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    spatial_state = {}
    for key, value in state_dict.items():
        if key.startswith("spatial_transformer"):
            spatial_state[key.replace("spatial_transformer.vit.", "")] = value

    temporal_state = {}
    for key, value in state_dict.items():
        if key.startswith("temporal_transformer") or key.startswith("classifier"):
            temporal_state[key] = value

    spatial_encoder.load_state_dict(spatial_state, strict=True)
    temporal_encoder.model.load_state_dict(temporal_state, strict=True)
    return spatial_encoder, temporal_encoder


def export_spatial_encoder(model, output_path, image_size, opset):
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["frame"],
        output_names=["embedding"],
        dynamic_axes={
            "frame": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
    )


def export_temporal_encoder(model, output_path, window_length, embed_dim, opset):
    dummy_embeddings = torch.randn(1, window_length, embed_dim, dtype=torch.float32)
    dummy_mask = torch.zeros(1, window_length, dtype=torch.bool)
    torch.onnx.export(
        model,
        (dummy_embeddings, dummy_mask),
        output_path,
        input_names=["embeddings", "padding_mask"],
        output_names=["logits"],
        dynamic_axes={
            "embeddings": {0: "batch"},
            "padding_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )


def write_metadata(output_dir, checkpoint_path, model_config):
    metadata = {
        "classes": CLASSES,
        "checkpoint_path": str(checkpoint_path),
        "image_size": int(model_config["image_size"]),
        "window_length": int(model_config["max_seq_length"]),
        "embed_dim": 768,
        "normalize": True,
        "pad_value": [123, 117, 104],
        "spatial_model": "spatial_encoder.onnx",
        "temporal_model": "temporal_encoder.onnx",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = dict(config["model"])
    model_config["num_classes"] = len(CLASSES)

    checkpoint_path = Path(args.checkpoint or config["evaluation"]["checkpoint"]).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = checkpoint_path.with_suffix("")
        output_dir = output_dir.parent / f"{output_dir.name}_onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    spatial_encoder, temporal_encoder = build_models(model_config, checkpoint_path, device)

    export_spatial_encoder(
        spatial_encoder,
        output_dir / "spatial_encoder.onnx",
        image_size=model_config["image_size"],
        opset=args.opset,
    )
    export_temporal_encoder(
        temporal_encoder,
        output_dir / "temporal_encoder.onnx",
        window_length=model_config["max_seq_length"],
        embed_dim=768,
        opset=args.opset,
    )
    write_metadata(output_dir, checkpoint_path, model_config)

    print(f"ONNX bundle written to {output_dir}")


if __name__ == "__main__":
    main()
