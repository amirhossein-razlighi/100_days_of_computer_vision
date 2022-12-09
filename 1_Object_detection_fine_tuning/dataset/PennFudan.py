from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os


class PennFudanDataset(Dataset):
    def __init__(self, root, transformations) -> None:
        super().__init__()
        self.root = root
        self.transforms = transformations

        self.images = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, index):
        image_path = os.path.join(self.root, "PNGImages", self.images[index])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = (mask == obj_ids[:, None, None])

        boxes = []
        number_of_objs = len(obj_ids)
        for i in range(number_of_objs):
            position = np.where(masks[i])
            xmin, ymin = np.min(position[1]), np.min(position[0])
            xmax, ymax = np.max(position[1]), np.max(position[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((number_of_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        is_crowd = torch.zeros((number_of_objs, ), dtype=torch.int64)

        target = dict()
        target['image_id'] = image_id
        target['is_crowd'] = is_crowd
        target['masks'] = masks
        target['area'] = area
        target['labels'] = labels
        target['boxes'] = boxes

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)
