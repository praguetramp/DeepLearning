import cv2
from matplotlib import pyplot as plt
from machlineLearn import haarcascades
img = cv2.imread("face.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("face ", gray_img)
cv2.waitKey()

haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Final_detected_Face :", cv2.COLOR_BGR2RGB(img))
cv2.waitKey()
