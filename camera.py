import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, a = cap.read()
    while ret:
        ret, a = cap.read()
        a = cv2.resize(a,(480,256))
        cv2.imshow("camera", a)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
