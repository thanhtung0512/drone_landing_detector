import torch 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
img_path = "data_train/images/train/img_train_400.jpg"
model = torch.hub.load('Ultralytics/yolov5', 'custom',
                            "F:/result_model/best.pt", force_reload=True,
                            trust_repo=True)
results = model(img_path)
df = results.pandas().xyxy[0]
print(df.head())
x1 = results.pandas().xyxy[0]['xmin'].values[0]
y1 = results.pandas().xyxy[0]['ymin'].values[0]
x2 = results.pandas().xyxy[0]['xmax'].values[0]
y2 = results.pandas().xyxy[0]['ymax'].values[0]

print(x1, x2 ,y1, y2)

img = cv.imread(img_path)
width = img.shape[1]
height = img.shape[0]
img = cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
cv.imshow("img",img)
cv.waitKey(0)


