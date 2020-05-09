import cv2
import numpy as np

def getBottomMidPoint(box):
    """Returns the x, y coordinate of the bottom center point of a bounding box.

    Args:
    - box: an array where box[0] is the top left y point
                          box[1] is the top left x point
                          box[2] is the bottom right y point
                          box[3] is the bottom right x point

    Returns:
    - tuple of the coordinates, ordered as (x, y)
    """
    xcoord = (box[1] + box[3]) / 2
    ycoord = box[2]
    return (int(xcoord), int(ycoord))


def getCenterPoint(box):
    """Returns the x, y coordinate of the center point of a bounding box.

    Args:
    - box: an array where box[0] is the top left y point
                          box[1] is the top left x point
                          box[2] is the bottom right y point
                          box[3] is the bottom right x point

    Returns:
    - tuple of the coordinates, ordered as (x, y)
    """
    xcoord = (box[1] + box[3]) / 2
    ycoord = (box[0] + box[2]) / 2
    return (int(xcoord), int(ycoord))


def checkSocialDistance(p1, p2, transform_mat, dist_thresh):
    """Checks if the distance between two points after a transformation is greater
    than a specified distance threshold.

    Args:
    - p1: first point
    - p2: second point
    - transform_mat: transformation matrix
    - dist_thresh: distance threshold

    Returns:
    - true if further than distance threshold, false otherwise
    """
    points = np.array([p1, p2])

    transformed = cv2.perspectiveTransform(np.float32([points]), transform_mat)[0]
    t1 = transformed[0]
    t2 = transformed[1]

    distance = (((t1[0] - t2[0]) ** 2) + ((t1[1] - t2[1]) ** 2)) ** 0.5

    return distance > dist_thresh
