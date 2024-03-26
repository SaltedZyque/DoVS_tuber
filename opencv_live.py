import cv2 as cv
import matplotlib.pyplot as plt

imgPath = "assets/test_3.jpeg"
img = cv.imread(imgPath)

#print(img.shape)
# cv.imshow("Display Window", img)
# k = cv.waitKey(0)

face_classifier = cv.CascadeClassifier(
	cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_cap = cv.VideoCapture(0)

def det_bounding_box(vid):
    g_vid = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(
        g_vid, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
    )

    # Create Bounding Box
    for (x, y, w, h) in face:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return face

#cv.imshow('Face Detect', img)
#k = cv.waitKey(0)

while True:
    result, video_frame = video_cap.read() # read video frames
    if not result:
        print("No Camera Detected")
        break

    faces = det_bounding_box(video_frame)

    cv.imshow(
        "Face Detection - Live Video Test", video_frame
    )

    if cv.waitKey(1) & 0xFF == ord("q"): # press "q" to exit
        break

video_cap.release()
cv.destroyAllWindows()
