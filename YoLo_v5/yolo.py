import torch
import cv2 as cv
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# imgs = ['/home/gahyeon/github/MDE-Object-Detection-Fusion/LapDepth-release/pretrained_img/img.jpg']

# results = model(imgs)

# results.show()

cap = cv.VideoCapture(0)

while(True):
    ret, cam = cap.read()

    if(ret) :
        imgs = cam

        results = model(imgs)
        print(type(results))
        print(results)

        cv.imshow('camera', results)
        
        if cv.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break

cap.release()
cv.destroyAllWindows()