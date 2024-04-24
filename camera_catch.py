from triangulation import catch_face
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

while True:
    key = cv2.waitKey(5)
    if key in (ord("q"), ord('й')):
        break
    success, img = cap.read()
    img = cv2.flip(img, 1)
    catch_face(img)
    cv2.imshow("Image", img)
    del img