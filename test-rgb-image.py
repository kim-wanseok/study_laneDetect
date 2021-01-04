# -*- coding: utf-8 -*-

import cv2
import numpy as np

def region_of_interest(img, vertices):
    mark = np.zeros_like(img)
    
    if len(img.shape) > 2:
        color = (255,255,255)
    else:
        color = 255

    cv2.fillPoly(mark, vertices, color)
    roi_image = cv2.bitwise_and(img, mark)
    return roi_image

def mark_img(img, blue_threshold=200, green_threshold=200, red_threshold=200, gray_threshold=200):
    if len(img.shape) > 2:
        bgr_threshold = [blue_threshold, green_threshold, red_threshold]
        thresholds = (img[:,:,0] < bgr_threshold[0]) | (img[:,:,1] < bgr_threshold[1]) | (img[:,:,2] < bgr_threshold[2])
        mark[thresholds] = [0,0,0]
    else:
        thresholds = (img[:,:] < gray_threshold)
        mark[thresholds] = 0
    return mark

def overlay_img(img, mark, color3=(255,255,255), color1=255):
    if len(mark.shape) > 2:
        thresholds = (mark[:,:,0] > 0) & (mark[:,:,1] > 0) & (mark[:,:,2] > 0)
    else:
        thresholds = (mark[:,:] > 0)
    if len(img.shape) > 2:
        img[thresholds] = color3
    else:
        img[thresholds] = color1
    return img

image = cv2.imread('./images/solidWhiteCurve.jpg')
height, width = image.shape[:2]
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

vertices = np.array([[(50,height), (width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)

# roi_image = region_of_interest(image, vertices)
roi_image = region_of_interest(gray_img, vertices)

mark = np.copy(roi_image)
mark = mark_img(mark, blue_threshold=200, green_threshold=200, red_threshold=200, gray_threshold=200)

image = overlay_img(image, mark, color3=(0,0,255), color1=128)
gray_img = overlay_img(gray_img, mark, color3=(0,0,255), color1=10)

cv2.imshow('result', image)
cv2.imshow('gray',gray_img)
cv2.imshow('roi', roi_image)
cv2.imshow('roi_white', mark)
cv2.waitKey(0)