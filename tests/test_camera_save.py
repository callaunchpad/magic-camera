import argparse
import cv2
from PIL import Image


def main(camera, file):
    """ Main entry point of the app """
    vid = cv2.VideoCapture(camera) 
    ret, frame = vid.read() 

    # flip color channels because of cv2 / PIL mismatch
    img = Image.fromarray(frame[:,:,::-1])
    img.save(file)

    vid.release()  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)
    parser.add_argument("--file", "-f", type=str, default="img.png")

    args = parser.parse_args()
    main(**vars(args))
