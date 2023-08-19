import cv2
import mrcnn.model as modellib
import os
import numpy as np

from mrcnn import visualize
import os

from src.Configs import InferenceConfig
from src.utils import refine_masks, resize_image, to_rle

IMAGE_SIZE = 512
DATA_DIR = os.path.join("custom data")
IMAGES_DIR = os.path.join(DATA_DIR, "test", "resized")
ROOT_DIR = os.path.join("")


label_names = ["lower_clothes", "upper_clothes"]

model_path = os.path.join("clothes20230221T1455", "mask_rcnn_clothes_0008.h5")

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '', "Provide path to trained weights"

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Convert data to run-length encoding
sub_list = []
missing_count = 0

images = os.listdir(IMAGES_DIR)

for image in images:
    image_path = str(os.path.join(IMAGES_DIR, image))

    if(not os.path.exists(image_path)):
        print("\nDoes not exist => ", image_path)
        continue

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = model.detect([resize_image(image_path)])
    r = result[0]

    thresh = 0.85

    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            if(r["scores"][m] < thresh):
                continue

            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                        (img.shape[1], img.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

        y_scale = img.shape[0] / IMAGE_SIZE
        x_scale = img.shape[1] / IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']

    visualize.display_instances(
        img,
        rois, masks, r['class_ids'],
        ['bg']+label_names,
        r['scores'],
        title=image, figsize=(12, 12)
        )
