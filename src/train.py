import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dataloader import get_dataloaders
from model import SimpleCNN
from trainer import train
from torchinfo import summary


def resolve_config_path(config_path):
    path = Path(config_path).expanduser()
    if path.is_file():
        return path

    fallback = Path(__file__).resolve().parent / path.name
    if path.name == "config.yaml" and fallback.is_file():
        return fallback

    raise FileNotFoundError(f"No se encontro el archivo de configuracion: {path}")


def load_config(config_path):
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now", lambda pattern: datetime.now().strftime(pattern)
        )
    cfg = OmegaConf.load(resolve_config_path(config_path))
    OmegaConf.resolve(cfg)
    return cfg


def get_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(name)


def run_training(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = get_device(cfg.device)
    # 1 Get dataloaders
    loaders, in_channels, num_classes = get_dataloaders(cfg)

    # 2 Create model
    model = SimpleCNN(in_channels, num_classes).to(device)
    summary(
        model,
        input_size=(1, in_channels, cfg.dataset.image_size, cfg.dataset.image_size),
        device=device.type,
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
    )

    # 3 Create loss function
    loss_fn = nn.CrossEntropyLoss()

    # 4 Create optimizer
    if cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=tuple(cfg.training.betas),
        )
    else:
        raise ValueError("training.optimizer debe ser 'sgd' o 'adam'")

    print(f"device: {device}")
    print(f"dataset: {cfg.dataset.name}, classes: {num_classes}")

    train(model, loaders, loss_fn, optimizer, cfg, device)


def cli_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    run_training(load_config(args.config))


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    if any(arg.startswith("--config") for arg in sys.argv[1:]):
        cli_main()
    else:
        main()  # pyright: ignore[reportCallIssue]
