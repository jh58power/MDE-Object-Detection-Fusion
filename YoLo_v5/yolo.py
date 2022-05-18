from multiprocessing import cpu_count
import re
from unittest import result
import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# imgs = '/home/gahyeon/github/MDE-Object-Detection-Fusion/LapDepth-release/pretrained_img/img.jpg'

# results = model(imgs)
rgb = (255, 0, 0)
# pred = results.pandas().xyxy[0]
xmin, ymin, xmax, ymax = [], [], [], []

# img = cv2.imread(imgs, cv2.IMREAD_COLOR)

cap = cv2.VideoCapture(0)

while(True):
    ret, cam = cap.read()

    if(ret) :
        
        results = model(cam)
        pred = results.pandas().xyxy[0]

        for i in range(len(pred)):
            xmin, ymin, xmax, ymax = int(pred.xmin[i]), int(pred.ymin[i]), int(pred.xmax[i]), int(pred.ymax[i])
            results = cv2.rectangle(cam, (xmin, ymin), (xmax, ymax), rgb, 2)

            cv2.imshow('camera', results)
        
        if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break
                     
# for i in range(len(pred)):
#     xmin, ymin, xmax, ymax = int(pred.xmin[i]), int(pred.ymin[i]), int(pred.xmax[i]), int(pred.ymax[i])
#     img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), rgb, 2)

# cv2.imshow('a',img)
# cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()


