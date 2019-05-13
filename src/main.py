from thymio import Thymio

import rospy
import cv2

video = Thymio('thymio10')

while not rospy.is_shutdown():
    rospy.sleep(1.)