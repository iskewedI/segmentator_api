import json
import cv2
import mrcnn.model as modellib
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from mrcnn import visualize
import os

from src.Configs import InferenceConfig
from src.utils import refine_masks, resize_image, to_rle

IMAGE_SIZE = 512
DATA_DIR = os.path.join("custom data")
ROOT_DIR = os.path.join("")

with open(os.path.join(DATA_DIR, "data.json")) as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]

model_path = os.path.join("clothes20230221T1049", "mask_rcnn_clothes_0008.h5")

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Prediction
# glob_list = glob.glob(f'/kaggle/working/fashion*/mask_rcnn_fashion_FILL.h5')
# model_path = glob_list[0] if glob_list else ''


sample_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
sample_df.head()

# Convert data to run-length encoding
sub_list = []
missing_count = 0

for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image_path = os.path.join(DATA_DIR, 'train', row['ImageId'])

    if(not os.path.exists(image_path)):
        print("\nDoes not exist => ", image_path)
        continue

    image = resize_image(image_path)

    result = model.detect([image])[0]

    if result['masks'].size > 0:
        masks, _ = refine_masks(result['masks'], result['rois'])
        for m in range(masks.shape[-1]):
            mask = masks[:, :, m].ravel(order='F')
            rle = to_rle(mask)
            label = result['class_ids'][m] - 1
            sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), row["Height"], row["Width"], label])
    else:
        # The system does not allow missing ids, this is an easy way to fill them
        sub_list.append([row['ImageId'], '1 1', row["Height"], row["Width"], 23])
        missing_count += 1

# The submission file is created, when all predictions are ready.
submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
print("Total image results: ", submission_df['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_df.head()

# Convert to CSV
submission_df.to_csv("submission.csv", index=False)


for i in range(10):
    image_id = sample_df.sample()['ImageId'].values[0]
    image_path = str(os.path.join(DATA_DIR, 'train', image_id))

    if(not os.path.exists(image_path)):
        print("\nDoes not exist => ", image_path)
        continue

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = model.detect([resize_image(image_path)])
    r = result[0]

    thresh = 0.9

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
        title=image_id, figsize=(12, 12)
        )
