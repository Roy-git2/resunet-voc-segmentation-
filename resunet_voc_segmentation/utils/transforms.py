# Albumentation-based image transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, shift_limit=0.1, rotate_limit=0, p=1),
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, test_transform

