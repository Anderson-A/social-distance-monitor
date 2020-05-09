import argparse
from pathlib import Path
import sys

import cv2
import imutils
import numpy as np

from birds_eye_view import getTransformation
from utils import getBottomMidPoint, getCenterPoint, checkSocialDistance
from nn_detector import Net


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to video")
ap.add_argument('-o', '--output', required=True, help='path or name to output video to. Recommended to be an avi file')
ap.add_argument('-m', '--model', required=True, help='path to model graph')
ap.add_argument('-t', '--thresh', required=True, help='model threshold for score of prediction (0.6 for ssd, 0.7 for faster-rcnn')
args = ap.parse_args()

# Verify paths
video_dir = Path(args.video)
if not video_dir.is_file():
    sys.exit('Cannot find provided video')

model_dir = Path(args.model)
if not model_dir.is_file():
    sys.exit('Cannot find provided model')

# Open video, get properties
cap = cv2.VideoCapture(str(video_dir))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare video to output
# dimensions for resizing input, done to reduce processing time and increase accuracy
resize_width = min(960, frame_width)
resize_height = round((resize_width / frame_width) * frame_height)
outVideo = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), framerate, (resize_width, resize_height))

# Get first frame of video to use for specifying homography, reset video capture
# to beginning
_, frame1 = cap.read()
cap.release()
cap = cv2.VideoCapture(str(video_dir))

# Resize frame to the size we'll be processing at
frame1 = imutils.resize(frame1, width=resize_width)

# Get transformation matrix and distance threshold from computation using user 
# inputs
transform_mat, dist_thresh = getTransformation(frame1)

# Initialize model
model = Net(args.model, float(args.thresh))

while cap.isOpened():
    # grab next frame of video
    ret, image = cap.read()

    # check if a frame was read successfully
    if ret:
        # resize image to reduce detection time and increase accuracy
        image = imutils.resize(image, width=resize_width)

        # detect people in the image
        boxes = model.processFrame(image)

        # make a list of the bottom center coordinate of each person's bounding
        # box. This represents the point where they are standing
        bottom_points = [getBottomMidPoint(box) for box in boxes]

        # make a list of the center coordinate of each person's bounding box.
        # If two people violate social distancing, we will draw a line
        # between their two centers.
        center_points = [getCenterPoint(box) for box in boxes]

        for i in range(len(boxes)):
            box1 = boxes[i]
            safe = True # To check if person was too close in at least one case
            for j in range(len(boxes)):
                if i == j:
                    continue
                # Check if these two people are social distancing
                sd = checkSocialDistance(bottom_points[i], bottom_points[j], transform_mat, dist_thresh)
                if not sd:
                    safe = False
                    image = cv2.line(image, center_points[i], center_points[j], (0,0,255), 1)
            if safe:
                image = cv2.rectangle(image, (box1[1],box1[0]), (box1[3],box1[2]), (0,255,0), 2)
            else:
                image = cv2.rectangle(image, (box1[1],box1[0]), (box1[3],box1[2]), (0,0,255), 2)

        # write image with annotations to the output video
        outVideo.write(image)

    else:
        # reached end of video
        break

cap.release()
outVideo.release()
