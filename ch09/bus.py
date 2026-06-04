from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo26n_ncnn_model", task="detect")

# Perform object detection on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Visualize the results
for result in results:
    result.show()