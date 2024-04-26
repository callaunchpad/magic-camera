import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

from cog import BasePredictor, Input, Path
from openai import OpenAI
import os
import base64
from io import BytesIO

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

def preprocess(image):
  img = Image.open(image)
  return img

class Predictor(BasePredictor):
  def setup(self):
    """Load the model into memory to make running multiple predictions efficient"""
    model_id = "masterful/gligen-1-4-generation-text-box"
    self.pipe = StableDiffusionGLIGENPipeline.from_pretrained(
      model_id, variant="fp16", torch_dtype=torch.float16
    )
    self.pipe.to("cuda")
    #os.getenv("OPENAI_API_KEY") # TODO
    #self.client = OpenAI()

  def gpt_prompt_boxes(self, description, cropped_img):
    buffered = BytesIO()
    cropped_img.save(buffered, format="JPEG") # TODO
    encoded_image = base64.b64encode(buffered.getvalue())
    messages = [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Here is an image with the description {description}. Describe the image to make a prompt for a stable diffusion model."
            },
            {
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{encoded_image}"
              }
            }
          ]
        }
      ]
    
    return self.client.chat.completions.create(
        model="gpt-4-1106-vision-preview",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

  def gpt_prompt_caption(self, descriptions):
    messages = [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"Here are several descriptions of objects: {', '.join(descriptions)}.\nCombine the descriptions to make a prompt for an image-generating stable diffusion model."
          }
        ]
      }
    ]
    return self.client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

  def create_prompt(self, image, boxes, phrases):
    new_phrases = []
    for (box, description) in zip(boxes, phrases):
      cropped = image.crop(box)
      new_phrases.append(self.gpt_prompt_boxes(description, cropped)) # TODO

    prompt = self.gpt_prompt_caption(new_phrases)

    return prompt, new_phrases

  # The arguments and types the model takes as input
  def predict(self,
        image: Path = Input("file path"), 
        num_inference_steps: int = 50,
  ) -> Path:
    """Run a single prediction on the model"""

    #--- DECONSTRUCTION PHASE ---#
    # Build the detector and download our pretrained weights
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

    im = cv2.imread(image)

    # Run detic model
    height, width, channels = im.shape
    outputs = predictor(im)
    phrases = [metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]
    boxes_uncut = outputs["instances"].pred_boxes.tensor.tolist()
    boxes = []
    for box_uncut in boxes_uncut:
      box = [box_uncut[0]/width, box_uncut[1]/height, box_uncut[2]/width, box_uncut[3]/height]
      boxes.append(box)
    prompt = "a cartoon scene"

    #--- RECONSTRUCTION PHASE ---#
    #processed_image = preprocess(image)
    # TODO take in processed_image
    #boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
    #phrases = ["sign", "mango"]
    #prompt, gpt_phrases = self.create_prompt(processed_image, boxes, phrases) # TODO

    output = self.pipe(
      prompt=prompt,
      gligen_phrases=phrases,
      gligen_boxes=boxes,
      gligen_scheduled_sampling_beta=1,
      output_type="pil",
      num_inference_steps=num_inference_steps,
    ).images
    new_path = "./out.jpg" 
    output[0].save(new_path)
    return Path(new_path)
