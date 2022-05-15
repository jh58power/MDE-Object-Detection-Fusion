import torch
import numpy as np
import cv2
# import pafy
from time import time


def load_model():
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def score_frame(frame):
        # frame: 단일 프레임; numpy/list/tuple 형식
        # return: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        model.to(device)
        frame = [frame]
        results = model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord
def class_to_label(x):
        # x 숫자 레이블 -> 문자열 레이블로 반환
        return classes[int(x)]
    
    
def plot_boxes(results, frame):
        # 경계상자와 레이블을 프레임에 플로팅
        # results: 프레임에서 모델이 감지한 객체의 레이블과 좌표
        # frame: 점수화된 프레임
        # return: 경계 상자와 레이블이 플로팅된 프레임
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, class_to_label(labels[i])
                            + ': ' + str(x1) + ', ' + str(x2) + ', ' + str(y1) + ', ' + str(y2),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame
    
    
model = load_model()
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, a = cap.read()
    while ret:
        ret, a = cap.read()

        result = score_frame(a)
        a = plot_boxes(result,a)
        cv2.imshow("camera", a)
        # 이미지를 보여주는 방식과 같습니다.
 
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # 종료 커맨드.
 
cap.release()
cv2.destroyAllWindows()