import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold): # low_threshold : high_threshould = 1:2 or 1:3
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mark = np.zeros_like(img)
    
    if len(img.shape) > 2:
        color = (255,255,255)
    else:
        color = 255

    cv2.fillPoly(mark, vertices, color)
    roi_image = cv2.bitwise_and(img, mark)
    return roi_image

def draw_lines(img, lines, color=(0,0,255), thickness=2):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_linesP(img, lines, color=[0,0,255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshould):
    lines = cv2.HoughLines(img, rho, theta, threshould)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hough_linesP(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_linesP(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0. ):
    return cv2.addWeighted(initial_img, α, img, β, λ)

cap = cv2.VideoCapture('./videos/solidWhiteRight.mp4')

while(cap.isOpened()):
    ret, image = cap.read()
    height, width = image.shape[:2]
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 70, 140)
    vertices = np.array([[(50,height), (width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    roi_img = region_of_interest(canny_img, vertices)

    hough_img = hough_linesP(roi_img, 1, 1*np.pi/180, 30, 10, 30)
    result = weighted_img(hough_img, image)

    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()