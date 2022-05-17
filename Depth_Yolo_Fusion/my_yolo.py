import cv2
# import pafy
from y_utils import load_model, score_frame, plot_boxes

    
yolo_model = load_model()


cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, a = cap.read()
    while ret:
        ret, a = cap.read()
        a = cv2.resize(a,(480,256))
        result = score_frame(yolo_model,a)
        a = plot_boxes(yolo_model,result,a)
        
        cv2.imshow("camera", a)
        # 이미지를 보여주는 방식과 같습니다.
 
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # 종료 커맨드.
 
cap.release()
cv2.destroyAllWindows()