import argparse
import cv2


def main(camera):
    """ Main entry point of the app """
    vid = cv2.VideoCapture(camera) 
  
    while True:
        ret, frame = vid.read() 
        cv2.imshow('frame', frame) 

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release()  
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
