# Drone’s landing area detector’s document

# Group Number 11

## Members

- Nguyễn Vũ Thanh Tùng (Team leader)
- Nguyễn Thành Đạt
- Nguyễn Tiến Hải
- Vũ Xuân Quyết

## How to run

1. Install requirements: 
`pip install -r requirements.txt`
2. Run main.py
`python main.py`

## Explain landing_detector’s implement

- In this project, we used OpenCV DNN to load trained yolov5s model under .onnx version and use it to detect coordinate of the bounding box for drone’s landing area.
- The following is the details steps for implementation of landing_detector.
    1. **DEFINE CONSTANTS**
        - **INPUT_WIDTH** and **INPUT_HEIGHT** are for the BLOB size. The **BLOB** 
        stands for Binary Large Object. It contains the data in readable raw format. The image must be converted to a blob so the network can process it. In our case, it is a 4D array object with the shape (1, 3, 640, 640)
        
        ```java
        INPUT_WIDTH = 640
        INPUT_HEIGHT = 640
        SCORE_THRESHOLD = 0.5 // To filter low probability class scores.
        NMS_THRESHOLD = 0.45 // To remove overlapping bounding boxes.
        CONFIDENCE_THRESHOLD = 0.45 // Filters low probability detections.
         
        # Text parameters.
        FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.7
        THICKNESS = 1
         
        # Colors.
        BLACK  = (0,0,0)
        BLUE   = (255,178,50)
        YELLOW = (0,255,255)
        ```
        
    2. ****PRE-PROCESSING YOLOv5 Model****
        - The method **pre**–**process** takes the image and the network as two arguments. Firstly, the image is converted to a blob, then it is set as input to the network. The function **`getUnconnectedOutLayerNames()`** provides the names of the output layers. It has features of all the layers, through which the image is forward propagated to acquire the detections. After processing, it returns the detection results.
        
        ```python
        def pre_process(self, input_image, net):
                # Create a 4D blob from a frame.
                blob = cv2.dnn.blobFromImage(
                    input_image, 1/255.0,  (self.INPUT_HEIGHT, self.INPUT_WIDTH), [0, 0, 0], 1, crop=False)
        
                # Sets the input to the network.
                net.setInput(blob)
        
                # Run the forward pass to get output of the output layers.
                outputs = net.forward(net.getUnconnectedOutLayersNames())
                return outputs
        ```
        
    3. ****POST-PROCESSING YOLOv5 Prediction Output****
        - In the previous function **`pre_process`**, we get the detection results as an object. It needs to be unwrapped for further processing. Before discussing the code any further, let us see the shape of this object and what it contains.
        - The returned object is a 2-D array. The output depends on the size of the input. For example, with the default input size of 640, we get a 2D array of size 25200×6 (rows and columns). The rows represent the number of detections. So each time the network runs, it predicts 25200 bounding boxes. Every bounding box has a 1-D array of 6 entries that tells the quality of the detection. This information is enough to filter out the desired detections.
        
        
        <p align="center">
<img src="https://imgur.com/QzwQ966">
</p>
        
        - The first two places are normalized center coordinates of the detected bounding box. Then comes the normalized width and height. Index 4 has the confidence score that tells the probability of the detection being an object. last entry tell the class scores of ‘area’ object of our problem.
        - **A. Filter Good Detections given by YOLOv5 Models**
            - While unwrapping, we need to be careful with the shape. With OpenCV-Python 4.1.2.30, the object is a tuple of a 3-D array of size **1x row x column**. It should be **row x column**. Hence, the array is accessed from the zeroth index.
            - The network generates output coordinates based on the input size of the blob,  i.e. 640. Therefore, the coordinates should be multiplied by the resizing factors to get the actual output. The following steps are involved in unwrapping the detections.
                1. Loop through detections.
                2. Filter out good detections.
                3. Get the index of the best class score.
                4. Discard detections with class scores lower than the threshold value. 
                
                ```python
                def post_process(self, input_image, outputs):
                
                        # Lists to hold respective values while unwrapping.
                        class_ids = []
                        confidences = []
                        boxes = []
                        # Rows.
                        rows = outputs[0].shape[1]
                        image_height, image_width = input_image.shape[:2]
                        # Resizing factor.
                        x_factor = image_width / self.INPUT_WIDTH
                        y_factor = image_height / self.INPUT_HEIGHT
                        # Iterate through detections.
                        for r in range(rows):
                            row = outputs[0][0][r]
                            confidence = row[4]
                            # Discard bad detections and continue.
                            if confidence >= self.CONFIDENCE_THRESHOLD:
                                classes_scores = row[5]
                                # Get the index of max class score.
                                class_id = classes_scores
                                #  Continue if the class score is above threshold.
                                if (class_id > self.SCORE_THRESHOLD):
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
                ```
                
        - **B. Remove Overlapping Boxes Predicted by YOLOv5**
            - After filtering good detections, we are left with the desired bounding boxes. However, there can be multiple overlapping bounding boxes, which may look like the following.
            
          
            <blockquote class="imgur-embed-pub" lang="en" data-id="a/4AMf2Zj" data-context="false" ><a href="//imgur.com/a/4AMf2Zj"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
            
            
            
            - This is solved by performing Non-Maximum Suppression. The function **`NMSBoxes()`**takes a list of boxes, calculates **[IOU** (Intersection Over Union](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/), and decides to keep boxes depending on the **`NMS_THRESHOLD`**. Curious about how it works? Check out [Non Maximum Suppression](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/) to know more. The result after using NMS Algorithm:
            
        
            <blockquote class="imgur-embed-pub" lang="en" data-id="a/4AMf2Zj" data-context="false" ><a href="//imgur.com/a/4AMf2Zj"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
            
            
            ```python
            # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
                  indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                  for i in indices:
                        box = boxes[i]
                        left = box[0]
                        top = box[1]
                        width = box[2]
                        height = box[3]             
                        # Draw bounding box.             
                        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
                        # Class label.                      
                        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])             
                        # Draw label.             
                        draw_label(input_image, label, left, top)
                  return input_image
            ```
            
        

1. **C. Detect method**

- Finally, we load the model. Perform pre-processing and post-processing and get the coordinate of the bounding box.

```python
def detect(self, image):

        classes = ['area']

        # Load image.
        # frame = cv2.imread("data_train/images/train/img_train_385.jpg")
        # Give the weight files to the model and load the network using       them.
        modelWeights = "best_13mb.onnx"
        net = cv2.dnn.readNet(modelWeights)
        # Process image.
        detections = self.pre_process(image, net)

        detections_coor = self.post_process(image.copy(), detections)

        return detections_coor
```

### Thank you for taking the time to read the documentation describing the implementation of the code in Python for detecting the coordinates of the bounding box for a drone landing area. We hope this information has been helpful in understanding the approach and implementation of the code. If you have any further questions or need additional information, please feel free to ask.
