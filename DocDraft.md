> to be converted to the README

# Concept and Plan

Live facial tracking applied to a model rig - including features like head movements, eyes and speech. Perhaps also an emotion recognition program to change expressions.

Checklist:
- [ ] Static Image Facial Detection
- [ ] Live Face Tracking through Webcam 
	- [ ] Precise Facial Feature Tracking (opt) - Landmark Detection? 
	- [ ] Deep Learning
	- [ ] Optimization 
- [ ] Connect Tracking to a Rigged Model (2D)
	- [ ] Implement as 3D (opt)
- [ ] (Opt) Emotion Recognition 
- [ ] OBS Implementation


# OpenCV
https://www.datacamp.com/tutorial/face-detection-python-opencv

OpenCV is a popular computer vision library - and supports Python, which we will be using.
- Contains >2500 algorithms allowing users to perform tasks like face recognition and object detection
- For face detection - OpenCV has models that have already been trained and so I don't need to train an algorithm from scratch.


# Face Detection

## Haar Cascade
https://www.researchgate.net/publication/3940582_Rapid_Object_Detection_using_a_Boosted_Cascade_of_Simple_Features

https://machinelearningmastery.com/using-haar-cascade-for-object-detection/

> The idea behind this technique involves using a cascade of classifiers to detect different features in an image. These classifiers are then combined into one strong classifier that can accurately distinguish between samples that contain a human face from those that don’t.
## Python Code

**Display Image in Window**
```
cv2.imshow("Disp Window", img)
k = cv2.waitKey(0)
```
waitKey closes the picture when a keyboard key is pressed

**Report Image Dimensions**
`img.shape`
Array is 3D - (Height, Width, Channels)
For coloured images the colour channel is BGR (opposite of RGB)

**Greyscale Conversion**
`cv2.cvtColor(imagename, cv2.COLOR_BGR2GRAY)

**Load Classifier**
```
face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

The "haarcascade_frontalface_default.xml" classifier detects frontal faces in visual input
https://github.com/opencv/opencv/tree/master/data/haarcascades - other classifiers

**Face Detection**
```
face = face_classifier.detectMultiScale(
    g_img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
```

- detectMultiScale() - identify faces of different sizes in input image
- scaleFactor - scale down input image for algorithm to easier detect large faces
	- 1.1 reduces by img size by 10%
- minNeighbors - specifies number of neighbouring rectangles needed to be identified for object to be valid detection
	- small values (like 1) will result in many false positives, large values will lose out on true positives.
- minSize - minimum size of object to be detected, any faces smaller than that will not be detected

**Bounding Box**
```
for (x, y, w, h) in face:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
```

**Convert back to RGB and Show**
```
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(5,5))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

# Video Capture and Recognition

## Video Capture
Documentation on `cv.VideoCapture()`

