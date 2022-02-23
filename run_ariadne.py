import cv2
import os

# ariadne
from scripts.ariadne import AriadnePlus

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', required=True, help='relative path of the input image')
parser.add_argument('--num_segments', default=20, help='number superpixels segments')
parser.add_argument('--show', action = "store_true", help='show result computation')
args = parser.parse_args()

num_segments = int(args.num_segments)
show_result = bool(args.show)

main_folder = os.getcwd()

##################################
# Loading Input Image
##################################
img_path = os.path.join(main_folder, args.img_path)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (640,480)) # resize necessary for the network model

##################################
# Initializing class
##################################
ariadne = AriadnePlus(main_folder, num_segments)

##################################
# Run pipeline
##################################
rv = ariadne.runAriadne(img, debug=True)

if show_result: 
    ##################################
    # Show result
    ##################################
    cv2.imshow("img_input", img)
    cv2.imshow("img_final", rv["img_final"])
    cv2.imshow("img_mask", rv["img_mask"])
    cv2.waitKey(0)