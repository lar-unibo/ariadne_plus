#! /home/lar/miniconda3/envs/ariadneplus/bin/python
import numpy as np
from PIL import Image 
import cv2 

# ros
import rospy
import sensor_msgs
import rospkg
from cv_bridge import CvBridge

# services and messages
from ariadne_plus.msg import *

# ariadne
from scripts.ariadne import AriadnePlus

import actionlib


class Action(object):

    def __init__(self):

        rospack = rospkg.RosPack()
        rospack.list()
        self.main_folder = rospack.get_path('ariadne_plus')
        self.bridge = CvBridge()
        
        # CAMERA PARAMS
        self.camera_matrix = [float(value) for value in rospy.get_param('ariadne_action/camera_matrix').split(", ")]
        self.camera_matrix = np.array(self.camera_matrix).reshape(3,3)
        self.distort_vec = np.array([float(value) for value in rospy.get_param('ariadne_action/distort_vec').split(", ")])
        self.camera_height = int(rospy.get_param('ariadne_action/camera_height'))
        self.camera_width = int(rospy.get_param('ariadne_action/camera_width'))
        self.topic_camera = str(rospy.get_param('ariadne_action/topic_camera'))

        self.topic_camera = rospy.get_param('ariadne_action/topic_camera')
        self.topic_action = rospy.get_param('ariadne_action/topic_action')
        self.ariadne_as = actionlib.SimpleActionServer(self.topic_action, spline_tckAction, execute_cb=self.callback, auto_start=False)
        self.ariadne_as.start()
        print("Action [{}] created".format(self.topic_action))

        # ARIADNE CLASS
        self.num_superpixels = int(rospy.get_param('ariadne_action/num_superpixels'))
        self.type_model = str(rospy.get_param('ariadne_action/type_model'))
        self.ariadne = AriadnePlus(self.main_folder, self.num_superpixels, type_model = self.type_model)
        print("Initialized Ariadne+ with {} superpixels and {} model type".format(self.num_superpixels, self.type_model))




    def callback(self, goal):
        
        image = rospy.wait_for_message(self.topic_camera, sensor_msgs.msg.Image)
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        cv_image  = cv2.resize(cv_image, (self.camera_width, self.camera_height))
        cv_image = cv2.undistort(cv_image, self.camera_matrix, self.distort_vec)

        print("received image with shape: ", cv_image.shape)

        rv  = self.ariadne.runAriadne(cv_image, debug=True)
        
        res = spline_tckResult()
        res.mask_image = self.generateImage(rv["img_mask"]) 
        res.image = image 
        res.tck = self.generateSpline(rv["spline_msg"])
        self.ariadne_as.set_succeeded(res)


    def generateImage(self, img_np):
        img = Image.fromarray(img_np).convert("RGB") 
        msg = sensor_msgs.msg.Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = img.height
        msg.width = img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img.width
        msg.data = np.array(img).tobytes()
        return msg

    def generateSpline(self, spline_ariadne):
        msg = []
        for a in spline_ariadne:
            tck = spline_tck()
            tck.t = a[0]
            tck.cx = a[1]
            tck.cy = a[2]
            tck.k = a[3]
            msg.append(tck)
        return msg



if __name__ == '__main__':
    
    rospy.init_node('ariadne_service')
    s = Action()
    while not rospy.is_shutdown():
        rospy.spin()