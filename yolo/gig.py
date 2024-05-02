import cv2
import numpy as np
import os
from datetime import datetime
from pymongo import MongoClient
from gridfs import GridFS

# MongoDB Atlas configuration
uri = "mongodb+srv://kalp9639:Ka11426w12!@bottledata.e6oqcxv.mongodb.net/ClusterBottle?retryWrites=true&w=majority&appName=ClusterBottle"
client = MongoClient(uri)
db = client['bottledatabase']  
fs = GridFS(db)

# Specify the directory where your files are located
directory = "/Users/kalp/Documents/SNU/6th semester/CSD326 - Software Eng/Project/SE_Bottle detection project/yolo"
directory_img = "/Users/kalp/Documents/SNU/6th semester/CSD326 - Software Eng/Project/SE_Bottle detection project/bottle-detection/frontend/src/img"
# Load YOLOv3 weights and configuration
weights_path = os.path.join(directory, "yolov3.weights")
cfg_path = os.path.join(directory, "yolov3.cfg")

# Load YOLOv3 model
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load class labels
classes_path = os.path.join(directory, "yolov3.txt")
with open(classes_path, "r") as f:
    classes = f.read().strip().split("\n")

# Access the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    bottle_detected = False  # Flag to indicate if bottle is detected in the frame

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'bottle':  # Check if detected object is a bottle
                bottle_detected = True
                break
        if bottle_detected:
            break

    if bottle_detected:
        # Generate unique image name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"bottle_detection_{timestamp}.jpg"
        
        # Save the frame as an image
        cv2.imwrite(os.path.join(directory_img, image_name), frame)
        
        # Fetch current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Save metadata along with the image in MongoDB
        metadata = {
            "timestamp": timestamp,
            "date": current_date,
            "time": current_time
        }
        
        # Open the saved image and store it in MongoDB GridFS along with metadata
        with open(os.path.join(directory_img, image_name), 'rb') as img_file:
            fs.put(img_file, filename=image_name, **metadata)

        print(f"Image uploaded to MongoDB GridFS: {image_name}")

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()