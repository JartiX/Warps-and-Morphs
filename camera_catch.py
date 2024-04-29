from triangulation import catch_face, face_mask
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

mask = cv2.imread("faces/girl.jpg")

while True:
    key = cv2.waitKey(5)
    if key in (ord("q"), ord('й')):
        break
    success, img = cap.read()
    img = cv2.flip(img, 1)
    catch_face(img)
    cv2.imshow("Image", img)
    del img

while True:
    key = cv2.waitKey(5)
    if key in (ord("q"), ord('й')):
        break
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = face_mask(img, mask, 0.5)
    cv2.imshow("Image", img)
    del img

# # From video
# cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
# cap = cv2.VideoCapture("faces/face_video2.mp4")
# frames = []
# while True:
#     check, img = cap.read()
#     if not check:
#         break

#     img = face_mask(img, mask, 1)
#     frames.append(img)
#     cv2.imshow("Video", img)
#     cv2.waitKey(1)
# cv2.waitKey(0)