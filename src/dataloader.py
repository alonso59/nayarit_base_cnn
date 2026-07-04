from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from augmentations import train_augmentations, val_augmentations


STANDARD_DATASETS = {
    "mnist": (datasets.MNIST, 1, 10),
    "fashion_mnist": (datasets.FashionMNIST, 1, 10),
    "cifar10": (datasets.CIFAR10, 3, 10),
    "cifar100": (datasets.CIFAR100, 3, 100),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class DatasetView(Dataset):
    def __init__(self, dataset, transform, indices=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        x, y = self.dataset[index]
        return self.transform(x), y


class ImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        folders = [p.name for p in Path(directory).iterdir() if p.is_dir()]
        if not folders:
            raise FileNotFoundError(f"No hay carpetas de clases en {directory}")
        try:
            classes = sorted(folders, key=int)
        except ValueError as exc:
            raise ValueError(
                "Las carpetas de clases deben llamarse 0, 1, 2, ..."
            ) from exc

        expected = [str(i) for i in range(len(classes))]

        if classes != expected:
            raise ValueError(f"Las carpetas de clases deben ser: {', '.join(expected)}")

        return classes, {name: int(name) for name in classes}


class PredictionImageDataset(Dataset):
    def __init__(self, images_dir, transform):
        self.images_dir = Path(images_dir)
        self.transform = transform

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta: {self.images_dir}")

        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No hay imagenes en {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), path.stem


def check_same_classes(classes, dataset, split_name):
    if dataset.classes != classes:
        raise ValueError(
            f"{split_name} debe tener estas carpetas: {', '.join(classes)}"
        )


def split_train_val(dataset, train_transform, val_transform, cfg):
    num_folds = cfg.dataset.num_folds
    fold_index = cfg.dataset.fold_index

    if not 1 < num_folds <= len(dataset):
        raise ValueError(f"num_folds debe estar entre 2 y {len(dataset)}")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index debe estar entre 0 y {num_folds - 1}")

    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    fold_size = len(indices) // num_folds
    start = fold_index * fold_size
    end = len(indices) if fold_index == num_folds - 1 else start + fold_size

    val_indices = indices[start:end]
    train_indices = indices[:start] + indices[end:]
    return (
        DatasetView(dataset, train_transform, train_indices),
        DatasetView(dataset, val_transform, val_indices),
    )


def load_folder_datasets(cfg):
    root = Path(to_absolute_path(cfg.dataset.path))

    if (root / "train_val").exists():
        train_ds = ImageFolder(root / "train_val")
        val_ds = None
        classes = train_ds.classes
    else:
        train_ds = ImageFolder(root / "train")
        classes = train_ds.classes
        if (root / "val").exists():
            val_ds = ImageFolder(root / "val")
            check_same_classes(classes, val_ds, "val")
        else:
            val_ds = None

    class_names = list(cfg.dataset.get("class_names", []))
    if class_names and len(class_names) != len(classes):
        raise ValueError(f"dataset.class_names debe tener {len(classes)} nombres.")

    test_path = root / "test"
    test_ds = None
    if test_path.exists() and not (test_path / "images").exists():
        test_ds = ImageFolder(test_path)
    if test_ds is not None:
        check_same_classes(classes, test_ds, "test")

    return train_ds, val_ds, test_ds, 3, len(classes)


def get_dataloaders(cfg):
    name = cfg.dataset.name.lower()

    if name == "folder":
        train_base, val_base, test_base, in_channels, num_classes = (
            load_folder_datasets(cfg)
        )
    else:
        if name not in STANDARD_DATASETS:
            valid_names = list(STANDARD_DATASETS) + ["folder"]
            raise ValueError(f"dataset.name debe ser uno de: {', '.join(valid_names)}")

        DatasetClass, in_channels, num_classes = STANDARD_DATASETS[name]
        root = to_absolute_path(cfg.dataset.root)
        train_base = DatasetClass(root, train=True, download=cfg.dataset.download)
        val_base = None
        test_base = DatasetClass(root, train=False, download=cfg.dataset.download)

    train_transform = train_augmentations(cfg.dataset.image_size, in_channels)
    val_transform = val_augmentations(cfg.dataset.image_size, in_channels)
    if val_base is None:
        train_ds, val_ds = split_train_val(
            train_base, train_transform, val_transform, cfg
        )
    else:
        train_ds = DatasetView(train_base, train_transform)
        val_ds = DatasetView(val_base, val_transform)
    test_ds = DatasetView(test_base, val_transform) if test_base is not None else None

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
        )
        if test_ds
        else None,
    }
    return loaders, in_channels, num_classes


def get_prediction_dataloader(cfg):
    name = cfg.dataset.name.lower()
    if name != "folder":
        raise ValueError("predict.py espera dataset.name=folder y dataset/test/images/")

    _, _, _, in_channels, num_classes = load_folder_datasets(cfg)
    transform = val_augmentations(cfg.dataset.image_size, in_channels)
    images_dir = Path(to_absolute_path(cfg.dataset.path)) / "test" / "images"
    dataset = PredictionImageDataset(images_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    return loader, in_channels, num_classes
