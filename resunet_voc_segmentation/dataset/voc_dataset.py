# Pascal VOC dataset loading and processing
import cv2
import numpy as np
from torchvision.datasets import VOCSegmentation

VOC_COLORMAP = [ [0, 0, 0], [128, 0, 0], ..., [0, 64, 128] ]  # Complete from your original list

class PascalVOCSearchDataset(VOCSegmentation):
    def __init__(self, root="~/data/pascal_voc", image_set="train", download=True, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)
        self.transform = transform

    def _convert_to_segmentation_mask(self, mask):
        height, width = mask.shape[:2]
        seg_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for idx, color in enumerate(VOC_COLORMAP):
            seg_mask[:, :, idx] = np.all(mask == color, axis=-1).astype(float)
        return seg_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].argmax(dim=2).squeeze()
        return image, mask
