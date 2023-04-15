import os
import torch
import cv2
import numpy as np


class LandingDetector:

    # Constants.




    
    # Colors.
  



    def __init__(self):

        self.name = "LandingDetector"


    def draw_label(self,im, label, x, y):
        """Draw text onto image at location."""
        # Get text size.
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a (0,0,0) rectangle.
        cv2.rectangle(im, (int(x), int(y)), (int(
            x + dim[0]), int(y + dim[1] + baseline)), (255, 0, 0), cv2.FILLED);
        # Display text inside the rectangle.
        cv2.putText(im, label, (int(x), int(
            y + dim[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)


    def pre_process(self,input_image, net):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            input_image, 1/255.0,  (640, 640), [0, 0, 0], 1, crop=False)

        # Sets the input to the network.
        net.setInput(blob)

        # Run the forward pass to get output of the output layers.
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs


    def post_process(self,input_image, outputs):
    
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # Resizing factor.
        x_factor = image_width / 640
        y_factor = image_height / 640
        # Iterate through detections.
        for r in range(rows):
                row = outputs[0][0][r]
                confidence = row[4]
                # Discard bad detections and continue.
                if confidence >= 0.45:
                    classes_scores = row[5:]
                    # Get the index of max class score.
                    class_id = np.argmax(classes_scores)
                    #  Continue if the class score is above threshold.
                    if (classes_scores[class_id] > 0.5):
                            confidences.append(confidence)
                            class_ids.append(class_id)
                            cx, cy, w, h = row[0], row[1], row[2], row[3]

                            x1 = int((cx - w/2) * x_factor)
                            y1 = int((cy - h/2) * y_factor)

                            x2 = int((cx + w/2) * x_factor)
                            y2 = int((cy + h/2) * y_factor)

                            # input_image = cv2.rectangle(input_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                            width = int(w * x_factor)
                            height = int(h * y_factor)
                            box = np.array([x1, y1, width, height])
                            boxes.append(box)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 0.45, 0.45)
        
        x1,y1 = 0,0
        x2,y2 = 0,0

        for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                # Draw bounding box.
                cv2.rectangle(input_image, (left, top),
                            (left + width, top + height), (255, 0, 0), 1)
                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height
               
        return (x1,y1,x2,y2)


    def detect(self, image):
           
        classes = ['area']

        # Load image.
        frame = cv2.imread("data_train/images/train/img_train_385.jpg")
        # Give the weight files to the model and load the network using       them.
        modelWeights = "best.onnx"
        net = cv2.dnn.readNet(modelWeights)
        # Process image.
        detectionss = self.pre_process(image, net)
        
        detections = self.post_process(image.copy(), detectionss)
            
        return detections