import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageOps
import sys

def get_model():
    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def process_image(image_path):
    # Load and process the image
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Convert PIL image to a tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)
    return image_tensor

def predict(model, image_tensor):
    with torch.no_grad():
        prediction = model([image_tensor])
    return prediction

def main(image_path, device="cuda:1"):
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", safety_checker=None)
    pipe.to(device)
    pipe.enable_attention_slicing()

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    print('rgb')

    model = get_model()
    model.to(device)  # Ensure the model is on the correct device
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.to(device)  # Move tensor to the correct device
    print('tensors')
    prediction = predict(model, image_tensor)

    # Print predictions to see what's detected
    print(prediction)

    prompt = "make the person's hair blonde"
    print(prompt)
    result_image = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1).images[0]
    print(prompt)
    # Display or save the result image as needed
    result_image.show()  # This will display the image; you might want to save it instead
    result_image.save("/home/ubuntu/dhruv/magic-camera/tests/result_image.jpg")
    print(f"Result image saved to /home/ubuntu/dhruv/magic-camera/tests")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py image_path [device]")
        sys.exit(1)
    image_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"
    main(image_path, device)
