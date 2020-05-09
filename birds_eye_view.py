import cv2
import numpy as np
import imutils

# List of reference points that the user selects
refPts = []

# Global reference to image user is selecting the points for
image = None

# Global reference for unmodified input image
clean_img = None

# global reference for image and transformation side by side
combined = None

# Global reference for transformation matrix
T_matrix = np.zeros((3,3))

# Global reference to pressed key
keypress = 0

# Global reference to points selected to form the six foot long line
linePts = []

# Instruction strings
four_points = 'Select 4 points to form a rectangle on the image by double clicking. \
The four points must be selected in order of bottom right, top right, top left, bottom left. \
Once four points have been selected, press c to confirm. \
Try to pick points along street lines for best results.'
check_transformation = 'Confirm that the homography looks good. Ideally the street \
lines in the original image should be aligned with the grid on the right. \
To reselect a point, press press either 1, 2, 3, or 4 to reselect the bottom right, \
top right, top left, or bottom left points respectively by double clicking. \
Once you are satisfied, press the c key to confirm.'
specify_dist = 'Select two points on the original photo by double clicking. \
These points should specify a line of which corresponds to about six feet \
(which is the CDCs recommended social distancing distance). Press c to confirm \
and r to restart.'


def drawPoints(event, x, y, flags, param):
    """ Callback function for when user initially picks the four points.

    Args:
    - event: event from openCV
    - x: x position of where event happened
    - y: y position of where event happened
    - flags: flags from openCV
    - param: parameters from openCV
    """
    # grab reference to the global variables
    global refPts
    global image

    # if the left mouse button was double clicked, record the (x, y) 
    # coordinates, draw circle at that point. Draw lines between points 
    # drawn after another.
    if event == cv2.EVENT_LBUTTONDBLCLK:

        if len(refPts) == 0:
            # draw first point
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            refPts.append((x, y))
            cv2.imshow("Select reference points", image)

        elif len(refPts) < 4:
            # connect current point and previous point
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.line(image, refPts[len(refPts) - 1], (x, y), (255, 0, 255), 2)
            refPts.append((x, y))

            # connect last and first points
            if len(refPts) == 4:
                cv2.line(image, refPts[0], (x, y), (255, 0, 255), 2)

            cv2.imshow("Select reference points", image)


def computeMatrix():
    """Computes the homography on the image using the selected points and
    transforms the image

    Returns:
    - the 3x3 transformation matrix
    """
    # grab reference to the global variables
    global refPts
    global image

    # points should be in order of bottom right, top right, top left, bottom left
    # x point first, then y point
    quadrilateral = np.array(refPts, dtype=np.float32)
    br, tr, tl, bl = quadrilateral

    # width of rectangle to transform to is max distance between either top left
    # and top right coordinates, or bottom left and bottom right coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # height of rectangle to transform to is max distance between either top left
    # and bottom left, or top right and bottom right
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # rectangle to transform to, the corresponding points should be in the same
    # order as the passed in points
    dst = np.array([
        [maxWidth - 1, maxHeight- 1],
        [maxWidth - 1, 0],
        [0, 0],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    homography = cv2.getPerspectiveTransform(quadrilateral, dst)

    # Find where the corners go after the transformation
    img_height, img_width = image.shape[:2]
    corners = np.array([
        [0,0],
        [0, img_height - 1],
        [img_width - 1, img_height - 1],
        [img_width - 1, 0]
    ], dtype=np.float32)

    # transformed corners
    t_corners = cv2.perspectiveTransform(np.float32([corners]), homography)[0]

    # Find the bounding rectangle around the warped image
    bx, by, bwidth, bheight = cv2.boundingRect(t_corners)

    # Compute new homography that makes the bounding box around the warped image
    # start at (0, 0)
    pure_translation = np.array([
        [1, 0, -bx],
        [0, 1, -by],
        [0, 0, 1]
    ])

    # if A and B are homographies, A*B represents the homography that applies
    # B first, then A
    M = np.matmul(pure_translation, homography)

    # Get the warped image
    warped = cv2.warpPerspective(image, M, (bwidth, bheight))
    # Resize the warped image to be same height as original so we can display
    # them side by side
    warped = imutils.resize(warped, height=img_height)

    return M, warped


def getTransformImg(warpedImg):
    gridOverlay = warpedImg.copy()
    for i in range(30, gridOverlay.shape[1], 30):
        gridOverlay = cv2.line(gridOverlay, (i, 0), (i, gridOverlay.shape[0]), (0, 255, 0), 1)
        if i < gridOverlay.shape[0]:
            gridOverlay = cv2.line(gridOverlay, (0, i), (gridOverlay.shape[1], i), (0, 255, 0), 1)
    
    return np.hstack((image, gridOverlay))


def redrawPoints(idx, x, y):
    """ Redraws the four points with lines between them.

    Args:
    - idx: index of point to change
    - x: x position of new point
    - y: y position of new point
    """
    global refPts
    global image
    global clean_img

    image = clean_img.copy()

    refPts.pop(idx)
    refPts.insert(idx, (x, y))

    for i, pt in enumerate(refPts):
        if i == 3:
            cv2.circle(image, pt, 5, (0, 255, 0), -1)
            cv2.line(image, pt, refPts[0], (255, 0, 255), 2)
        else:
            cv2.circle(image, pt, 5, (0, 255, 0), -1)
            cv2.line(image, pt, refPts[i+1], (255, 0, 255), 2)


def redoPoint(event, x, y, flags, param):
    """ Callback function for when user reselects a point.

    Args:
    - event: event from openCV
    - x: x position of where event happened
    - y: y position of where event happened
    - flags: flags from openCV
    - param: parameters from openCV
    """
    global refPts
    global T_matrix
    global combined

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if keypress == 1:
            redrawPoints(0, x, y)
            T_matrix, warped = computeMatrix()
            combined = getTransformImg(warped)
            cv2.imshow('Check transformation', combined)

        elif keypress == 2:
            redrawPoints(1, x, y)
            T_matrix, warped = computeMatrix()
            combined = getTransformImg(warped)
            cv2.imshow('Check transformation', combined)

        elif keypress == 3:
            redrawPoints(2, x, y)
            T_matrix, warped = computeMatrix()
            combined = getTransformImg(warped)
            cv2.imshow('Check transformation', combined)

        elif keypress == 4:
            redrawPoints(3, x, y)
            T_matrix, warped = computeMatrix()
            combined = getTransformImg(warped)
            cv2.imshow('Check transformation', combined)


def drawLine(event, x, y, flags, param):
    """Callback function for when a user draws the line.

    Args:
    - event: event from openCV
    - x: x position of where event happened
    - y: y position of where event happened
    - flags: flags from openCV
    - param: parameters from openCV
    """
    global linePts
    global combined

    if event == cv2.EVENT_LBUTTONDBLCLK:

        if len(linePts) == 0:
            # draw first point
            cv2.circle(combined, (x, y), 5, (0, 255, 0), -1)
            linePts.append((x, y))
            cv2.imshow("Draw six foot long line", combined)

        elif len(linePts) == 1:
            # draw second point and form line
            cv2.circle(combined, (x, y), 5, (0, 255, 0), -1)
            cv2.line(combined, linePts[0], (x, y), (255,0,255), 2)
            linePts.append((x, y))
            cv2.imshow("Draw six foot long line", combined)


def getTransformation(img):
    """Gets the transformation matrix that would transform the image to be in a
    birds eye view. Reference points to use in calculating the matrix are
    specified by the user. User also specifies a distance threshold for social
    distancing, should be around 6 feet

    Args:
    - img: image to find the transformatino matrix for a birds eye view

    Returns:
      The 3x3 transformation matrix
      The number of pixels AFTER transformation that correspond to 6 feet
    """
    # grab reference to global image, set it
    global image
    image = img

    # Clone an unmodified copy of the image
    global clean_img
    clean_img = img.copy()

    global combined
    global T_matrix

    # grab reference to pressed key
    global keypress

    # grab reference to user specified line
    global linePts

    # Create named window that we will display on while the user picks the points
    cv2.namedWindow('Select reference points')
    cv2.setMouseCallback('Select reference points', drawPoints)

    # Show directions
    print(four_points)

    while True:
        # display image and wait for event
        cv2.imshow('Select reference points', image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'c' key is pressed and all coordinates have been entered,
        # break from the loop
        if key == ord("c") and len(refPts) == 4:
            break

    # No longer need the window
    cv2.destroyAllWindows()

    T_matrix, warped = computeMatrix()

    combined = getTransformImg(warped)

    # Create window that we will display for user to confirm the transformation
    cv2.namedWindow('Check transformation')
    cv2.setMouseCallback('Check transformation', redoPoint)

    # Show directions
    print(check_transformation)

    while True:
        # display image and wait for event
        cv2.imshow('Check transformation', combined)
        key = cv2.waitKey(1) & 0xFF

        # if one of keys 1 to 4 is pressed, record it so mouse callback 
        # can know which point to redo
        if key == ord("1"):
            keypress = 1
        elif key == ord('2'):
            keypress = 2
        elif key == ord('3'):
            keypress = 3
        elif key == ord('4'):
            keypress = 4
        elif key == ord("c"):
            break

    # No longer need the window
    cv2.destroyAllWindows()

    # Create named window that we will display on while the user picks the six foot long line
    cv2.namedWindow('Draw six foot long line')
    cv2.setMouseCallback('Draw six foot long line', drawLine)

    # Show directions
    print(specify_dist)

    clean_combined = combined.copy()

    while True:
        # display image and wait for event
        cv2.imshow('Draw six foot long line', combined)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset line
        if key == ord('r'):
            linePts.clear()
            combined = clean_combined.copy()
        elif key == ord('c') and len(linePts) == 2:
            break

    cv2.destroyAllWindows()

    points = np.array([linePts[0], linePts[1]])
    transformed = cv2.perspectiveTransform(np.float32([points]), T_matrix)[0]
    t1 = transformed[0]
    t2 = transformed[1]

    dist_thresh = (((t1[0] - t2[0]) ** 2) + ((t1[1] - t2[1]) ** 2)) ** 0.5

    return T_matrix, int(dist_thresh)
