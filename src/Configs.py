from mrcnn.config import Config

class ClothesConfig(Config):
    NUM_CATS = 2
    IMAGE_SIZE = 512
    NAME = "Clothes"
    NUM_CLASSES = NUM_CATS + 1

    GPU_COUNT = 1
    IMAGES_PER_GPU = 6

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 130

    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 70

class InferenceConfig(ClothesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
