import cv2
import numpy as np

# Load YOLOv5 model
model_path = "best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

# Load image
image_path = "data_train/images/train/img_train_400.jpg"
image = cv2.imread(image_path,cv2.IMREAD_COLOR)
image = image.astype(np.float32)

# Resize image to 640x640
image = cv2.resize(image, (640, 640))

# Prepare input blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop = False)

# Set the input blob to the network
net.setInput(blob)

# Forward pass
# output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward()

# Get class labels
class_names = ['area']  # Define class names, in this case, only 'area' is used

# Loop through the outputs and extract object detections
objects = []
for output in outputs:
    
    for detection in output:
        
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id] if class_id < len(scores) else 0.0  # Set confidence to 0.0 if class_id is out of bounds
        if confidence > 0.9 and class_names[class_id] == 'area':  # Filter by class name and confidence threshold
            
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            x_min = int(center_x - width//2)
            y_min = int(center_y - height//2)
            x_max = int(center_x + width//2)
            y_max = int(center_y + height//2)
            objects.append((x_min, y_min, x_max, y_max))
            print((x_min, y_min, x_max, y_max))

# Draw bounding boxes on the image
for obj in objects:
    x_min, y_min, x_max, y_max = obj
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(image, 'area', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display the resulting image

