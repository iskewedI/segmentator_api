import cv2
import mrcnn.model as modellib
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import keras.backend as K_Backend

from mrcnn import visualize
import os

from src.Configs import InferenceConfig
from src.utils import MAT2b64, b642MAT, refine_masks

IMAGE_SIZE = 512
DATA_DIR = os.path.join("custom data")
IMAGES_DIR = os.path.join(DATA_DIR, "test", "resized")
ROOT_DIR = os.path.join("")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 3.5gb of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3584)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

label_names = ["bg", "lower_clothes", "upper_clothes"]

model_path = os.path.join("clothes20230221T1455", "mask_rcnn_clothes_0008.h5")

sess = tf.compat.v1.Session()
# model.make_predict_function()
graph = tf.compat.v1.get_default_graph()

set_session(sess)

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Convert data to run-length encoding
sub_list = []
missing_count = 0

def predict(img_b64):
    global sess
    global graph

    api_result = {
        "masked_imgb64": None,
        "maskb64": None,
        "mask_coords": None,
        "classes_detected": None
    }

    with graph.as_default():
        set_session(sess)

        image = b642MAT(img_b64)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        result = model.detect([image])
        r = result[0]

        thresh = 0.85
        labels_result = []

        if r['masks'].size > 0:
            masks = np.zeros((image.shape[0], image.shape[1], r['masks'].shape[-1]), dtype=np.uint8)

            for m in range(r['masks'].shape[-1]):
                if(r["scores"][m] < thresh):
                    continue

                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                            (image.shape[1], image.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)

                labels_result.append(r["class_ids"][m])

            y_scale = image.shape[0] / IMAGE_SIZE
            x_scale = image.shape[1] / IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

            masks, rois = refine_masks(masks, rois)
        else:
            return None

        masks_only = image.copy()

        masks_only[:] = 0

        for i in range(masks.shape[-1]):
            visualize.apply_mask(masks_only, masks[:, :, i], (1, 1, 1), 1)

            # Apply in the image too, so we can see the masks in top of the image
            visualize.apply_mask(image, masks[:, :, i], (0, 0, 0), 1)

        api_result["maskb64"] = MAT2b64(cv2.cvtColor(masks_only, cv2.COLOR_BGR2RGB))
        api_result["masked_imgb64"] = MAT2b64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        api_result["classes_detected"] = ",".join(label_names[classid] for classid in labels_result)

    K_Backend.clear_session()
    return api_result
