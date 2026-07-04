from torchvision import transforms


def train_augmentations(image_size, in_channels):
    """
    TODO:
    Ejemplos de transformaciones disponibles. 
    Elegir adecuadamente, combinarlas y ajustar probabilidades y factores de procesamiento.

    Generales:
    transforms.Resize((image_size, image_size))
    transforms.ToTensor()
    transforms.Normalize(mean, std)

    Geometricas:
    transforms.RandomHorizontalFlip(p=0.5)
    transforms.RandomVerticalFlip(p=0.5)
    transforms.RandomRotation(degrees=(-10, 10))
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))

    Intensidad/color:
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    transforms.RandomGrayscale(p=0.1)
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
    transforms.RandomAutocontrast(p=0.5)
    transforms.RandomEqualize(p=0.5)

    Degradacion:
    transforms.RandomPosterize(bits=4, p=0.5)
    transforms.RandomSolarize(threshold=192, p=0.5)
    transforms.RandomInvert(p=0.5)
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5)

    Crop:
    transforms.RandomCrop(size=(image_size, image_size), padding=4)
    transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.33))

    """
    mean = [0.0] * in_channels
    std = [1.0] * in_channels

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # NO MODIFICAR
            # ====================================================================
            # TODO: Agrega aqui transformaciones aleatorias solo para entrenamiento.
            
            
            # ====================================================================
            transforms.ToTensor(),  # NO MODIFICAR
            transforms.Normalize(mean, std),  # NO MODIFICAR
        ]
    )


def val_augmentations(image_size, in_channels):
    # Aqui no se modifica la imagen, solo se normaliza y se convierte a tensor
    mean = [0.0] * in_channels
    std = [1.0] * in_channels

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # NO MODIFICAR
            transforms.ToTensor(),  # NO MODIFICAR
            transforms.Normalize(mean, std),  # NO MODIFICAR
        ]
    )
