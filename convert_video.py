# Takes a path to a folder of image sequences and saves them as a video

import argparse
import cv2
from pathlib import Path
import sys

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True, help='path to directory with image sequences')
ap.add_argument('-o', '--output', required=True, help='path or name to output video to. Recommended to be an avi file')
ap.add_argument('-f', '--frames', required=True, help='framerate of video')
args = ap.parse_args()

img_dir = Path(args.images)
if not img_dir.is_dir():
    sys.exit('Cannot find provided image directory')

# Get resolution for video
all_images = img_dir.glob('*.jpg')
size = cv2.imread(str(next(all_images)), cv2.IMREAD_GRAYSCALE).shape[::-1]
all_images = img_dir.glob('*.jpg')

# Prepare video to output
outVideo = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), int(args.frames), size)

# read images and write to output video
for filename in all_images:
    img = cv2.imread(str(filename), cv2.IMREAD_COLOR)
    outVideo.write(img)

outVideo.release()
