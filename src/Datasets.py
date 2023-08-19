import json
import os
import cv2
import numpy as np
from mrcnn import utils as mrcnn_utils
from src.utils import coco_find_img_by_id, resize_image

# Data directories
DATA_DIR = os.path.join("custom data")
ROOT_DIR = os.path.join("")

IMAGE_SIZE = 512

label_descriptions_path = os.path.join(DATA_DIR, "data.json")

# Load local data
with open(label_descriptions_path) as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

class ClothesDataset(mrcnn_utils.Dataset):
    def __init__(self, df, coco):
        super().__init__(self)

        self.coco = coco

        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("clothes_dataset", i + 1, name)

        # Add images
        for i, row in df.iterrows():
            self.add_image("clothes_dataset",
                           image_id=row.name,
                           path=str(os.path.join(DATA_DIR, 'train', row.name)),
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'],
                           width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x) - 1] for x in info['labels']] # indexs starts from 0

    def load_image(self, image_id):
        img = self.image_info[image_id]
        return resize_image(img['path'])

    def load_mask(self, internal_img_id):
        info = self.image_info[internal_img_id]

        labels = []

        img = coco_find_img_by_id(self.coco.imgs, info["id"])
        anns_ids = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(anns_ids)

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(anns)), dtype=np.uint8)

        for i, ann in enumerate(anns):
            height = int(info["height"])
            width = int(info["width"])
            category_id = int(ann["category_id"])

            coco_mask = self.coco.annToMask(ann)

            sub_mask = coco_mask.reshape((height, width), order='F')
            sub_mask = cv2.resize(coco_mask, (IMAGE_SIZE, IMAGE_SIZE))
            sub_mask = sub_mask.reshape((IMAGE_SIZE, IMAGE_SIZE), order='F')

            mask[:, :, i] = sub_mask

            labels.append(category_id)

        return mask, np.array(labels)
