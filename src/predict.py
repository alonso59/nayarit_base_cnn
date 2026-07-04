import argparse
import csv
from pathlib import Path

import torch

from dataloader import get_prediction_dataloader
from eval import resolve_checkpoint_path
from model import SimpleCNN
from train import get_device, load_config


def run_prediction(cfg, checkpoint_path, output_path):
    device = get_device(cfg.device)
    loader, in_channels, num_classes = get_prediction_dataloader(cfg)

    model = SimpleCNN(in_channels, num_classes).to(device)
    checkpoint_path = resolve_checkpoint_path(cfg, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "y_pred"])

        with torch.no_grad():
            for x, image_ids in loader:
                x = x.to(device)
                logits = model(x)
                y_pred = logits.argmax(dim=1).cpu().tolist()

                for image_id, pred in zip(image_ids, y_pred):
                    writer.writerow([image_id, pred])

    print(f"predictions: {output_path}")


def cli_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_prediction(cfg, args.checkpoint, args.output)


if __name__ == "__main__":
    cli_main()
