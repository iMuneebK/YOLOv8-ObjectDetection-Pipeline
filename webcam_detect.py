from ultralytics import YOLO
import cv2

def run_webcam():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Webcam detection ready. Uncomment run_webcam() to use.")
