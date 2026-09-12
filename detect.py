from ultralytics import YOLO
import cv2
import sys

def detect_image(img_path):
    model = YOLO('yolov8n.pt')
    results = model(img_path)
    for r in results:
        im_array = r.plot()
        cv2.imshow("Detection", im_array)
        cv2.waitKey(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_image(sys.argv[1])
    else:
        print("Usage: python detect.py <image_path>")
