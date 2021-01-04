# -*- coding: utf-8 -*-

import cv2
import numpy as np

image = cv2.imread('./images/solidWhiteCurve.jpg')
mark = np.copy(image) # 이미지 복사

blue_threshold = 200
green_threshold = 200
red_threshold = 200
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# BGR 제한 값보다 작으면 검은색으로 
thresholds = (image[:,:,0] < bgr_threshold[0]) | (image[:,:,1] < bgr_threshold[1]) | (image[:,:,2] < bgr_threshold[2])
# print(type(thresholds))
# print(thresholds)
# print (mark.shape)
# print(thresholds.shape)
mark[thresholds] = [0,0,0]

cv2.imshow('white', mark)
cv2.imshow('result', image)
cv2.waitKey(0)
