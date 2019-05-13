import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt

import rospy

import rosnode
import rostopic
import rosmsg

import std_msgs.msg
import sensor_msgs.msg
import nav_msgs.msg
from sensor_msgs.msg import Range, Image, CompressedImage

from PID import PID

class Thymio:
    
    def __init__(self, thymio_name):
        """init"""
        self.thymio_name = thymio_name

        print(self.thymio_name)

        self.frame = []

        rospy.init_node('hand_following_thymio_controller', anonymous=True)

        self.angular_pid = PID(Kd=5, Ki=0, Kp=0.5)
        self.linear_pid = PID(Kd=5, Ki=0, Kp=0.5)
        self.object_pid = PID(Kd=3, Ki=0, Kp=0.5)

        self.total_rectangle = 9
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.traverse_point = []

        self.camera_subscriber = rospy.Subscriber(self.thymio_name + '/camera/image_raw/compressed', CompressedImage, self.camera_callback_compressed, queue_size=1, buff_size=2**24)


    def draw_rect(self,frame):
        rows, cols, _ = frame.shape

        self.x1 = np.array(
            [6*rows/20, 6*rows / 20, 6* rows / 20, 9* rows/20, 9 * rows /20, 9 * rows/20, 12 * rows/20,
            12 * rows /20, 12 * rows /20], dtype=np.uint32)

        self.y1 = np.array(
            [9*cols/20, 10*cols / 20, 11* cols / 20, 9* cols/20, 10 * cols /20, 11 * cols/20, 9 * cols/20,
            10 * cols /20, 11 * cols /20], dtype=np.uint32)

        self.x2 = self.x1 + 10
        self.y2 = self.y1 + 10

        for i in range(self.total_rectangle):
            cv2.rectangle(frame, (self.y1[i], self.x1[i]),
                          (self.y2[i], self.x2[i]),
                          (0,255,0),1)
        
        return frame

    def hand_histogram(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([90,10,3], dtype = hsv_frame.dtype)

        for i in range(self.total_rectangle):
            roi[i*10:i*10+10,0:10] = hsv_frame[self.x1[i]:self.x1[i] + 10, self.y1[i]:self.y1[i] + 10]
        
        hand_hist = cv2.calcHist([roi], [0,1], None, [180,256], [0,180,0,256])
        return cv2.normalize(hand_hist, hand_hist, 0,255, cv2.NORM_MINMAX)
    
    def centroid(self, max_contour):
        moment = cv2.moments(max_contour)

        if moment["m00"] != 0:
            cx = int(moment["m10"]/moment["m00"])
            cy = int(moment["m01"]/moment["m00"])
            return cx, cy
        else:
            return None

    def hist_masking(self,frame,hist):
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0,1], hist,[0,180,0,256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
        cv2.filter2D(dst,-1,disc,dst)
        ret, thresh = cv2.threshold(dst, 150,255,cv2.THRESH_BINARY)

        thresh = cv2.merge((thresh, thresh, thresh))

        return cv2.bitwise_and(frame, thresh)

    def contours(self, hist_mask_image):
        gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
        cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return cont

    def max_contour(self,contour_list):
        max_i = 0
        max_area = 0
        
        for i in range(len(contour_list)):
            cnt = contour_list[i]

            area_cnt = cv2.contourArea(cnt)

            if area_cnt > max_area:
                max_area = area_cnt
                max_i = i
        
        return contour_list[max_i]

    def farthest_point(self, defects, contour, centroid):
        if defects is not None and centroid is not None:
            s = defects[:,0][:,0]
            cx, cy = self.centroid(contour)

            x = np.array(contour[s][:,0][:,0], dtype=np.float)
            y = np.array(contour[s][:,0][:,1], dtype=np.float)

            xp = cv2.pow(cv2.subtract(x,cx),2)
            yp = cv2.pow(cv2.subtract(y,cy),2)
            dist = cv2.sqrt(cv2.add(xp,yp))

            dist_max_i = np.argmax(dist)

            if dist_max_i < len(s):
                farthest_defect = s[dist_max_i]
                farthest_point = tuple(contour[farthest_defect][0])
                return farthest_point
            else:
                return None
    
    def draw_circles(self, frame, traverse_point):
        if traverse_point is not None:
            for i in range(len(traverse_point)):
                cv2.circle(frame, traverse_point[i], int(5 -(5*i*3)/100),[0,255,255],-1)


    def manage_image_opr(self, frame, hand_hist):
        hist_mask_image = self.hist_masking(frame, hand_hist)
        contour_list = self.contours(hist_mask_image)
        max_cont = self.max_contour(contour_list)

        cnt_centroid = self.centroid(max_cont)

        cv2.circle(frame, cnt_centroid, 5, [255,0,255], -1)

        if max_cont is not None:
            hull = cv2.convexHull(max_cont, returnPoints=False)
            defects = cv2.convexityDefects(max_cont,hull)
            far_point = self.farthest_point(defects, max_cont, cnt_centroid)
            print("Centroid : " + str(cnt_centroid)+ ", farthest Point : " + str(far_point))

            cv2.circle(frame, far_point, 5, [0,0,255], -1)
            if len(self.traverse_point) < 20:
                self.traverse_point.append(far_point)
            else:
                self.traverse_point.pop(0)
                self.traverse_point.append(far_point)
            
            self.draw_circles(frame, self.traverse_point)

    def camera_callback_compressed(self, data):
        compressed = data.data
        np_arr = np.fromstring(data.data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # # preprocessing
        # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray,(5,5),0)
        # ret, thres1 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # # find countours
        # contours, hierarchy = cv2.findContours(thres1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        rect = self.draw_rect(image)
        hist = self.hand_histogram(rect)
        self.manage_image_opr(rect,hist)
        # mask = self.hist_masking(rect, hist)
        # self.frame = image

        # max_area = 0
        # for i in rrange(len(contours)):
        #     cnt=contours[i]
        #     area = cv2.contourArea(cnt)
        #     if area > max_area:
        #         max_area = area
        #         ci = i
        
        # cnt= contours[ci]
        # hull = cv2.convexHull(cnt)
        
        cv2.imshow('image', rect)
        cv2.waitKey(1)
    
    def camera_stream(self):
        while not rospy.is_shutdown():
            print(self.frame)
            cv2.imshow('image', self.frame)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# if __name__ == '__main__':

#     thymio = Thymio("thymio10")
    # thymio.camera_stream()
    

# def print_msg_definition(msg_type):
#     print(rosmsg.get_msg_text(msg_type))

# print(rostopic.get_topic_type('/thymio10/camera/camera_info')[0])

# print_msg_definition('sensor_msgs/CameraInfo')


# for compr in ['', '/compressed', '/theora']:
#     topic = '/thymio10/camera/image_raw%s' % compr
#     msg_type = rostopic.get_topic_type(topic)[0]
#     print('%s: %s' % (topic, msg_type))

# # print_msg_definition('sensor_msgs/Image')


# data_msg = rospy.wait_for_message('/thymio10/camera/image_raw', sensor_msgs.msg.Image)
# format_msg = rospy.wait_for_message('/thymio10/camera/camera_info', sensor_msgs.msg.CameraInfo)

# pixels = np.fromstring(data_msg.data, dtype=np.dtype(np.uint8)).reshape(format_msg.height, format_msg.width, 3)

# plt.imshow(pixels)









# video capture

# cap = cv2.VideoCapture("/thymio10/camera/image_raw")

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()