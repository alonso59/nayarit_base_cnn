import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from dataloader import get_dataloaders
from model import SimpleCNN
from train import get_device, load_config


def resolve_checkpoint_path(cfg, checkpoint_path=None):
    configured_path = checkpoint_path
    if not configured_path and "eval" in cfg:
        configured_path = cfg.eval.get("checkpoint_path")
    if configured_path:
        checkpoint_path = Path(str(configured_path)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(to_absolute_path(str(checkpoint_path)))
    else:
        checkpoint_path = Path(to_absolute_path(str(Path(cfg.output_dir) / "model.pt")))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No se encontro el checkpoint: {checkpoint_path}\n"
            "Usa --checkpoint output/.../checkpoints/best.pt, "
            "configura eval.checkpoint_path=output/.../checkpoints/best.pt "
            "o usa output_dir=... para cargar output_dir/model.pt."
        )
    return checkpoint_path


def collect_predictions(model, loader, loss_fn, device):
    model.eval()
    losses, y_true, y_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(loss_fn(logits, y).item())
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    y_pred = y_prob.argmax(axis=1)
    return np.mean(losses), y_true, y_pred, y_prob


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def classification_metrics(cm):
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)

    return {
        "accuracy": tp.sum() / total,
        "precision_macro": precision.mean(),
        "sensitivity_macro": sensitivity.mean(),
        "specificity_macro": specificity.mean(),
        "f1_macro": f1.mean(),
    }


def roc_curve(y_true_binary, scores):
    order = np.argsort(scores)[::-1]
    y = y_true_binary[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp / max(tp[-1], 1)]
    fpr = np.r_[0, fp / max(fp[-1], 1)]
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


def auroc_macro(y_true, y_prob, num_classes):
    aucs = []
    for c in range(num_classes):
        y_binary = (y_true == c).astype(int)
        if y_binary.min() != y_binary.max():
            aucs.append(roc_curve(y_binary, y_prob[:, c])[2])
    return float(np.mean(aucs)) if aucs else float("nan")


def plot_confusion_matrix(cm, path, class_names=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicción")
    ax.set_ylabel("clase real")
    ax.set_title("Matriz de confusión")
    if class_names:
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks, class_names, rotation=45, ha="right")
        ax.set_yticks(ticks, class_names)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(y_true, y_prob, num_classes, path):
    import matplotlib.pyplot as plt

    y_one_hot = np.eye(num_classes)[y_true].ravel()
    scores = y_prob.ravel()
    fpr, tpr, auc = roc_curve(y_one_hot, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"micro AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="azar")
    ax.set_xlabel("1 - especificidad")
    ax.set_ylabel("sensibilidad")
    ax.set_title("Curva ROC micro-promedio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_evaluation(cfg, checkpoint_path=None):
    device = get_device(cfg.device)
    loaders, in_channels, num_classes = get_dataloaders(cfg)

    model = SimpleCNN(in_channels, num_classes).to(device)
    checkpoint_path = resolve_checkpoint_path(cfg, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"checkpoint: {checkpoint_path}")

    loader = loaders["val"]
    loss_fn = nn.CrossEntropyLoss()
    loss, y_true, y_pred, y_prob = collect_predictions(model, loader, loss_fn, device)

    cm = confusion_matrix(y_true, y_pred, num_classes)
    metrics = classification_metrics(cm)
    metrics["loss"] = loss
    metrics["auroc_macro"] = auroc_macro(y_true, y_prob, num_classes)

    eval_dir = Path(cfg.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    class_names = (
        list(cfg.dataset.class_names) if cfg.dataset.name.lower() == "folder" else None
    )
    plot_confusion_matrix(cm, eval_dir / "confusion_matrix.png", class_names)
    plot_roc(y_true, y_prob, num_classes, eval_dir / "roc_curve.png")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    with open(eval_dir / "metrics.txt", "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    print(f"eval logs: {eval_dir}")


def cli_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    run_evaluation(load_config(args.config), args.checkpoint)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    run_evaluation(cfg)


if __name__ == "__main__":
    if any(arg.startswith(("--config", "--checkpoint")) for arg in sys.argv[1:]):
        cli_main()
    else:
        main()  # pyright: ignore[reportCallIssue]
