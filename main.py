from imgaug import augmenters as iaa
from mrcnn import visualize
from pycocotools.coco import COCO
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import numpy as np
import os
import pandas as pd
import random

from src.Configs import ClothesConfig
from src.Datasets import ClothesDataset

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Data directories
DATA_DIR = os.path.join("custom data")
ROOT_DIR = os.path.join("")

train_csv_dir = os.path.join(DATA_DIR, "train.csv")

coco = COCO(os.path.join(DATA_DIR, "data.json"))
# Categories Ids to load
coco_cats = coco.loadCats((1, 2))

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks
NUM_CATS = 3
IMAGE_SIZE = 512
COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'

# Config
config = ClothesConfig()
config.display()

label_names = [label['name'] for label in coco_cats]

segment_df = pd.read_csv(train_csv_dir)
segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]

print("Total segments: ", len(segment_df))

# Gathering same info together
image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')
print("Total images: ", len(image_df))

# Visualize data randomly
dataset = ClothesDataset(image_df, coco)
dataset.prepare()

for i in range(10):
   image_id = random.choice(dataset.image_ids)
   image = dataset.load_image(image_id)
   mask, class_ids = dataset.load_mask(image_id)

   print(f"For image => {image_id}, class ids found => {class_ids}")
   print("Available classes => ", dataset.class_names)
   visualize.display_top_masks(
    image,
    mask,
    class_ids,
    dataset.class_names,
    limit=4
    )

# This code partially supports k-fold training,
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 2

kf = KFold(n_splits=N_FOLDS, random_state=3, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

# Splitting into train and val datasets
def get_fold():
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]

train_df, valid_df = get_fold()

train_dataset = ClothesDataset(train_df, coco)
train_dataset.prepare()

valid_dataset = ClothesDataset(valid_df, coco)
valid_dataset.prepare()

# See segments and report data
# train_segments = np.concatenate(train_df['CategoryId'].values).astype(int)
# print("Total train images: ", len(train_df))
# print("Total train segments: ", len(train_segments))

# plt.figure(figsize=(12, 3))
# values, counts = np.unique(train_segments, return_counts=True)
# plt.bar(values, counts)
# plt.xticks(values, label_names, rotation='vertical')
# plt.show()

# valid_segments = np.concatenate(valid_df['CategoryId'].values).astype(int)
# print("Total validation images: ", len(valid_df))
# print("Total validation segments: ", len(valid_segments))

# plt.figure(figsize=(12, 3))
# values, counts = np.unique(valid_segments, return_counts=True)
# plt.bar(values, counts)
# plt.xticks(values, label_names, rotation='vertical')
# plt.show()

# Training the model
LR = 1e-4
EPOCHS = [2, 6, 8]

model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

# Augmentate data
# augmentation = iaa.Sequential([
#     iaa.Fliplr(0.5), # only horizontal flip here
# ])

# Training the model, first head layers
model.train(train_dataset, valid_dataset,
            learning_rate=2e-3,
            epochs=EPOCHS[0],
            layers='heads',
            augmentation=None)

history = model.keras_model.history.history

# Training all layers
model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[1],
            layers='all',
            augmentation=None)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

# Reducing LR and training again
model.train(train_dataset, valid_dataset,
            learning_rate=LR/5,
            epochs=EPOCHS[2],
            layers='all',
            augmentation=None)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]


# Vsualize training history and choose the best epoch.
epochs = range(EPOCHS[-1])

plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()

plt.show()

# Getting best epoch
best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])


