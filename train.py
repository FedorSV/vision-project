from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model

# Train the model
results = model.train(data='E:\Project\datasets\drones\drones.yaml', epochs=100, imgsz=640)