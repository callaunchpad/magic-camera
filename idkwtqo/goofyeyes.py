# Install detectron2
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
# Use the below line to install detectron2 if the above one has an error
# clone and install Detic


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
import PIL
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'Detic/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
# Build the detector and download our pretrained weights
os.chdir(' Detic')
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)
# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)
# Download a sample image and display. Replace path here to try your own images!
import os
from google.colab import drive
drive.mount("/content/drive",force_remount=True)
USER_PATH = f"/content/drive/MyDrive/magic-camera/users/annie/"
DATA_PATH = "/home/ubuntu/magic-camera/idkwtqo/bookshelf.png"
# path = os.path.join(DATA_PATH, "IMG_2448.jpg")
im = cv2.imread(DATA_PATH)
# image = open(path, "rb")
# !wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
# im = cv2.imread("./desk.jpg")
# cv2_imshow(im)
# Run model and show results
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], metadata)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

googly_eyes = cv2.imread('/home/ubuntu/magic-camera/idkwtqo/googlyeyes.png', cv2.IMREAD_UNCHANGED)
image_area = im.shape[0] * im.shape[1]

if googly_eyes.shape[2] == 3:  # Ensure there's an alpha channel
    alpha_channel = np.ones((googly_eyes.shape[0], googly_eyes.shape[1]), dtype=googly_eyes.dtype) * 255
    googly_eyes = np.dstack([googly_eyes, alpha_channel])
eyes_alpha = googly_eyes[:, :, 3] / 255.0  # Normalize alpha channel

def get_eyes_region(box, eyes_aspect_ratio):
    x_center = box[0] + (box[2] - box[0]) / 2
    y_center = box[1] + (box[3] - box[1]) / 2
    width = (box[2] - box[0]) * 2/3  # 2/3 of the width of the bounding box
    height = width * eyes_aspect_ratio  # Maintain the aspect ratio of the googly eyes
    x_start = int(x_center - width / 2)
    y_start = int(y_center - height / 2)
    return x_start, y_start, int(width), int(height)

# Function to overlay image with alpha channel
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    x1, x2 = max(x, 0), min(x + img_overlay.shape[1], img.shape[1])
    y1, y2 = max(y, 0), min(y + img_overlay.shape[0], img.shape[0])
    x1o, x2o = max(-x, 0), min(img.shape[1] - x, img_overlay.shape[1])
    y1o, y2o = max(-y, 0), min(img.shape[0] - y, img_overlay.shape[0])
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1 - alpha
    for c in range(3):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c])

from google.colab.patches import cv2_imshow

# Load and predict an image
image_path = '/home/ubuntu/magic-camera/idkwtqo/bookshelf.png'
im = cv2.imread(image_path)

bboxes = outputs["instances"].pred_boxes.tensor.tolist()
eyes_aspect_ratio = googly_eyes.shape[0] / googly_eyes.shape[1]
min_area_threshold = image_area / 16

for box in bboxes:
    # Calculate the area of the detected object
    box_area = (box[2] - box[0]) * (box[3] - box[1])

    # Proceed only if the object area is greater than or equal to the threshold
    if box_area >= min_area_threshold:
        x, y, w, h = get_eyes_region(box, googly_eyes.shape[0] / googly_eyes.shape[1])
        resized_eyes = cv2.resize(googly_eyes[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
        resized_alpha = cv2.resize(googly_eyes[:, :, 3], (w, h), interpolation=cv2.INTER_AREA) / 255.0

        overlay_image_alpha(
            img_overlay=resized_eyes,
            pos=(x, y),
            alpha_mask=resized_alpha
        )
os.chdir('..')
