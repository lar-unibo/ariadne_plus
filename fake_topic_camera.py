#! /home/alessio/anaconda3/envs/ariadneplus/bin/python
import numpy as np
import cv2, os
from PIL import Image
# ros
import rospy
import sensor_msgs.msg
import rospkg

def generateImage(img_np):
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

if __name__ == '__main__':
    
    rospy.init_node('fake_topic_camera')
    r = rospy.Rate(30)

    rospack = rospkg.RosPack()
    rospack.list()
    main_folder = rospack.get_path('ariadne_plus')


    ##################################
    # Loading Input Image
    ##################################
    img_path = os.path.join(main_folder, rospy.get_param('ariadne_plus/img_path'))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,480)) # resize necessary for the network model
    
    img_msg = generateImage(img)

    pub = rospy.Publisher(rospy.get_param('ariadne_plus/topic_camera'), sensor_msgs.msg.Image, queue_size=1)

    while not rospy.is_shutdown():
        pub.publish(img_msg)
        r.sleep()
