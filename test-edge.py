import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold): # low_threshold : high_threshould = 1:2 or 1:3
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)



image = cv2.imread('./images/solidWhiteCurve.jpg')
height, width = image.shape[:2]
gray_img = grayscale(image)
blur_img = gaussian_blur(gray_img, 5)
canny_img = canny(blur_img, 70, 140)

cv2.imshow('result', canny_img)
cv2.waitKey(0)