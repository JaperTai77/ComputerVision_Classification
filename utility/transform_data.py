import torchvision.transforms as transforms

from core.config import Settings


def transform_data(augmentation:bool=False):
    if augmentation:
        data_transform = transforms.Compose([
            transforms.Resize(Settings.resize_shape, Settings.resize_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(Settings.resize_shape, Settings.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    return data_transform