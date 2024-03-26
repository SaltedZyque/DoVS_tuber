import cv2 as cv
import dlib
import numpy as np

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

# define function to draw points and shapes of landmarks
## there are alternative functions for drawing only specific parts of the face, check tutorial article

def draw_shape_points(np_shape, img):
    """Draws the shape using points for every landmark"""

    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)


def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates

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
        shape = predictor(g_vid, face)

        # convert to np array
        shape = shape_to_np(shape)

        # draw all shape points
        draw_shape_points(shape,video_frame)


    cv.imshow(
        "Landmark Detection - Live Video Test", video_frame
    )

    if cv.waitKey(1) & 0xFF == ord("q"): # press "q" to exit
        break

video_cap.release()
cv.destroyAllWindows()
