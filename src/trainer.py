from pathlib import Path

import torch
from tqdm import tqdm


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def train_step(model, loader, loss_fn, optimizer, device, desc="train"):
    model.train()
    total_loss, total_acc = 0, 0

    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


def val_step(model, loader, loss_fn, device, desc="val"):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_acc += accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


def train(model, loaders, loss_fn, optimizer, cfg, device):
    run_dir = Path(cfg.output_dir)
    plot_dir = run_dir / "plots"
    checkpoint_dir = run_dir / "checkpoints"
    plot_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = -1

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc = train_step(
            model, loaders["train"], loss_fn, optimizer, device, f"epoch {epoch} train"
        )
        val_loss, val_acc = val_step(
            model, loaders["val"], loss_fn, device, f"epoch {epoch} val"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        plot_history(history, plot_dir / "training.png")

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    if loaders["test"] is not None:
        test_loss, test_acc = val_step(model, loaders["test"], loss_fn, device, "test")
        print(f"test loss {test_loss:.4f} acc {test_acc:.4f}")

    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"logs: {run_dir}")


def plot_history(history, path):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(epochs, train_loss, "r-", label="train loss")
    ax_loss.plot(epochs, val_loss, "r--", label="val loss")
    ax_loss.set_ylabel("loss")
    ax_loss.set_ylim(0, max(max(train_loss + val_loss) * 1.1, 0.01))

    ax_acc.plot(epochs, train_acc, "b-", label="train acc")
    ax_acc.plot(epochs, val_acc, "b--", label="val acc")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0, 1)

    lines = [*ax_loss.get_lines(), *ax_acc.get_lines()]
    ax_loss.legend(handles=lines, loc="best")
    ax_loss.set_xlabel("epoch")
    ax_loss.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
