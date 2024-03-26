import cv2 as cv
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Can define whichever range of points we want in the dlib 68
JAWLINE_POINTS = list(range(0,17)) # 0-16
RIGHT_EYEBROW_POINTS = list(range(17, 22)) # 17-21
LEFT_EYEBROW_POINTS = list(range(22, 27)) # 22-26
NOSE_BRIDGE_POINTS = list(range(27, 31)) # 27-30
LOWER_NOSE_POINTS = list(range(31, 36)) # 31-35
RIGHT_EYE_POINTS = list(range(36, 42)) # 36-41
LEFT_EYE_POINTS = list(range(42, 48)) # 42-47
MOUTH_OUTLINE_POINTS = list(range(48, 61)) # 48-60
MOUTH_INNER_POINTS = list(range(61, 68)) # 61-67
ALL_POINTS = list(range(0, 68))

#imutils gives the index automatically
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Some other params
MIN_EAR = 0.18

# define function to draw points and shapes of landmarks
## there are alternative functions for drawing only specific parts of the face, check tutorial article

def draw_shape_points(np_shape, img, colour):
    """Draws the shape using points for every landmark"""

    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv.circle(img, (x, y), 2, colour, -1)

def draw_shape_points_range(np_shape, image, range, colour):
    """draws shape of points for range of landmarks specified"""

    np_shape_display = np_shape[range]
    draw_shape_points(np_shape_display, image, colour)

def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates

# define functions for identifying specific facial features
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1] , eye_points[5])
    B = dist.euclidean(eye_points[2] , eye_points[4])
    C = dist.euclidean(eye_points[0] , eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize model, using dlib 68 shape predictor and face detector
p = "py/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


video_cap = cv.VideoCapture(0)

# frame_width = int(video_cap.get(3))
# frame_height = int(video_cap.get(4))

# size = (frame_width, frame_height)

while True:
    result, video_frame = video_cap.read() # read video frames

    if not result:
        print("No Camera Detected")
        break

    # resize and convert grayscale
    video_frame = cv.resize(video_frame,(0,0),fx = 1 , fy = 1)
    g_vid = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)

    # detect faces -- this replaces the opencv face classifier
    faces = detector(g_vid, 0)

    # find landmark for each face
    for (i, face) in enumerate(faces):
        
        # alternate way to draw a bounding box (replaces x,y etc. with functions)
        cv.rectangle(video_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)

        # get shape with predictor
        fshape = predictor(g_vid, face)

        # convert to np array
        fshape = shape_to_np(fshape)

        left_eye = fshape[leftEyeStart:leftEyeEnd]
        right_eye = fshape[rightEyeStart:rightEyeEnd]
        
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)

        # draw all shape points
        draw_shape_points(fshape,video_frame, (0,255,0))

        if left_EAR < MIN_EAR:
            draw_shape_points_range(fshape,video_frame,LEFT_EYE_POINTS,(0,0,255))
            cv.putText(video_frame, "Left Eye Closed!", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if right_EAR < MIN_EAR:
            draw_shape_points_range(fshape,video_frame,RIGHT_EYE_POINTS,(0,0,255))
            cv.putText(video_frame, "Right Eye Closed!", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
    cv.imshow(
        "Landmark Detection - Live Video Test", video_frame
    )

    if cv.waitKey(1) & 0xFF == ord("q"): # press "q" to exit
        break

video_cap.release()
cv.destroyAllWindows()
