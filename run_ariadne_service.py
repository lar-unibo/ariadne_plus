#! /home/lar/miniconda3/envs/ariadneplus/bin/python
import numpy as np
from PIL import Image 

# ros
import rospy
import sensor_msgs
import rospkg
from cv_bridge import CvBridge

# services and messages
from ariadne_plus.srv import getSplines, getSplinesResponse
from ariadne_plus.msg import spline_tck

# ariadne
from scripts.ariadne import AriadnePlus


class Service(object):

    def __init__(self):

        rospack = rospkg.RosPack()
        rospack.list()
        self.main_folder = rospack.get_path('ariadne_plus')

        self.topic_service = rospy.get_param('ariadne_service/topic_service')
        self.topic_camera = rospy.get_param('ariadne_service/topic_camera')
        _ = rospy.Service(self.topic_service, getSplines, self.service_callback)
        print("service [{}] created".format(self.topic_service))

        # ARIADNE CLASS
        self.num_superpixels = int(rospy.get_param('ariadne_service/num_superpixels'))
        self.type_model = str(rospy.get_param('ariadne_service/type_model'))
        self.ariadne = AriadnePlus(self.main_folder, self.num_superpixels, type_model = self.type_model)
        print("Initialized Ariadne+ with {} superpixels and {} model type".format(self.num_superpixels, self.type_model))

        self.bridge = CvBridge()


    def service_callback(self, req):
        
        image = rospy.wait_for_message(self.topic_camera, sensor_msgs.msg.Image)
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        print("received image with shape: ", cv_image.shape)

        rv  = self.ariadne.runAriadne(cv_image, debug=True)
        
        res = getSplinesResponse()
        res.mask_image = self.generateImage(rv["img_mask"]) 
        res.tck = self.generateSpline(rv["spline_msg"])



        return res
 
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
    s = Service()
    while not rospy.is_shutdown():
        rospy.spin()