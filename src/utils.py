import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt
import base64
import PIL.Image as Image
import io

IMAGE_SIZE = 512

def resize_image(image_path):
    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img

# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

# RLE encoding/decoding
def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle

def img_to_RLE(img, bits=8, binary=True, view=True):
    """
    img: Grayscale img.
    bits: what will be the maximum run length? 2^bits
    """
    if binary:
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if view:
        figure=plt.figure(figsize=(10, 10))

        plt.imshow(img)
        plt.show()

    encoded = []
    shape=img.shape
    count = 0
    prev = None
    fimg = img.flatten()
    th=127
    for pixel in fimg:
        if binary:
            if pixel<th:
                pixel=0
            else:
                pixel=1
        if prev==None:
            prev = pixel
            count+=1
        else:
            if prev!=pixel:
                encoded.append((count, prev))
                prev=pixel
                count=1
            else:
                if count<(2**bits)-1:
                    count+=1
                else:
                    encoded.append((count, prev))
                    prev=pixel
                    count=1

    encoded.append((count, prev))

    return encoded

def RLE_decode(encoded, shape):
    decoded=[]

    for rl in encoded:
        r,p = rl[0], rl[1]
        decoded.extend([p]*r)

    dimg = np.array(decoded).reshape(shape)

    return dimg

def coco_find_img_by_id(coco_imgs, img_name):
    res = None

    for _, img in coco_imgs.items():
        if(img["file_name"] == img_name):
            res = img
            break

    return res

# B64 transformations
def MAT2b64(mat):
    return base64.b64encode(cv2.imencode('.jpg', mat)[1]).decode()

def b642MAT(b64str) -> np.ndarray:
    imgbytes = base64.b64decode(b64str)
    return np.array(Image.open(io.BytesIO(imgbytes)))

def b642img(b64str):
    imgdata = base64.b64decode(b64str)
    return Image.open(io.BytesIO(imgdata))
