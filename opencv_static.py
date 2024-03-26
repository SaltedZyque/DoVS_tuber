import cv2 as cv
import matplotlib.pyplot as plt

imgPath = "assets/test_3.jpeg"
img = cv.imread(imgPath)

#print(img.shape)
# cv.imshow("Display Window", img)
# k = cv.waitKey(0)

g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_classifier = cv.CascadeClassifier(
	cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    g_img, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
)

# Create Bounding Box
for (x, y, w, h) in face:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

#cv.imshow('Face Detect', img)
#k = cv.waitKey(0)

# Change back to RGB
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

'''
#cv.imshow("Display Window", img_rgb)
#k = cv.waitKey(0)
'''

plt.figure(figsize=(5,5))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
