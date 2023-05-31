import torch
import numpy as np
import cv2
from time import time

def load_model():
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def score_frame(model,frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        frame = [frame]
        results = model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        print(labels)
        print(cord)
        return labels, cord

def class_to_label(model,x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        classes = model.names
        return classes[int(x)]
    
    
def plot_boxes(model,results, ori_frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        labels, cord = results
        n = len(labels)
        frame = ori_frame.copy()
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        print("{:10}  {:4} {:4} {:4} {:4}".format("ClassName","X1","Y1","X2","Y2"))
        print("="*30)

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                center_point = [(x1+x2)/2,(y1+y2)/2]
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, class_to_label(model,labels[i])
                            + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
                print("{:10} {:4} {:4} {:4} {:4}".format(class_to_label(model,labels[i]),x1,y1,x2,y2))
                print()
        return frame

def depth_plot_boxes(model,results, ori_frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        labels, cord = results
        n = len(labels)
        frame = ori_frame.copy()
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                center_point = [(x1+x2)/2,(y1+y2)/2]
                # print(frame[0][0])
                # print(center_point)
                center_depth = frame[int(center_point[0])][int(center_point[1])]
                
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, class_to_label(model,labels[i])
                            + ' depth : ' + str(center_depth), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1)

        return frame
    
