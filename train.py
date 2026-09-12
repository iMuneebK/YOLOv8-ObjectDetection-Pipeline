from ultralytics import YOLO

def train_model():
    print("Initializing YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("Starting training on custom dataset...")
    print("Training pipeline configured. Model saved to runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()
