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

def draw_fit_line(img, lines, color=(0,0,255), thickness=10):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def hough_lines(img, rho, theta, threshould):
    lines = cv2.HoughLines(img, rho, theta, threshould)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    # return line_img
    return lines

def hough_linesP(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_linesP(line_img, lines)
    # return line_img
    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0. ):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def cal_slopeDegree(coord_x1, coord_y1, coord_x2, coord_y2):
    slope_degree = (np.arctan2(coord_y1 - coord_y2, coord_x1 - coord_x2) * 180 ) / np.pi
    return slope_degree

def filterSlope(line, slope, min_degree, max_degree):
    line_arr = line[np.abs(slope) > min_degree]
    line_arr = line[np.abs(slope) < max_degree]
    slope_degree = slope[np.abs(slope) > min_degree]
    slope_degree = slope[np.abs(slope) < max_degree]
    # line_arr = line[(np.abs(slope) < min_degree) and (np.abs(slope) > max_degree)]
    # slope_degree = slope[(np.abs(slope) < min_degree).all() and (np.abs(slope) > max_degree).all()]
    return line_arr, slope_degree

def get_fitline(img, f_lines):
    lines = np.squeeze(f_lines)
    # print(lines.shape)
    lines = lines.reshape(lines.shape[0]*2, 2)
    height, width = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = output[0], output[1], output[2], output[3]
    slope = vy / vx
    intercept = y0 - slope*x0

    # x1, y1 = (y1 - intercept) / slope, height
    # x2, y2 = (y2 - intercept) / slope, height
    y1, y2 = height, int(height/2+100)
    x1, x2 = int((y1 - intercept)/slope), int((y2 - intercept)/slope)
    # print(type(slope), type(intercept), type(x1), type(x2), type(y1), type(y2)) 
    result = [x1, y1, x2, y2]
    return result



# image = cv2.imread('./images/solidWhiteCurve.jpg')
image = cv2.imread('./images/slope_test.jpg')

height, width = image.shape[:2]
gray_img = grayscale(image)
blur_img = gaussian_blur(gray_img, 5)
canny_img = canny(blur_img, 70, 210)

vertices = np.array([[(50,height), (width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
roi_img = region_of_interest(canny_img, vertices)

# hough_img = hough_lines(roi_img, 1, 1*np.pi/180, 70)
line_arr = hough_linesP(roi_img, 1, 1*np.pi/180, 30, 20, 30)

line_arr = np.squeeze(line_arr)

slope_degree = cal_slopeDegree(line_arr[:,0], line_arr[:,1], line_arr[:,2], line_arr[:,3])

line_arr, slope_degree = filterSlope(line_arr, slope_degree, 95, 160)

L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
L_lines, R_lines = L_lines[:,None], R_lines[:,None]

temp = np.zeros_like(image, dtype=np.uint8)

L_fit_line = get_fitline(image, L_lines)
R_fit_line = get_fitline(image, R_lines)

draw_fit_line(temp, L_fit_line)
draw_fit_line(temp, R_fit_line)

result = weighted_img(temp, image)
cv2.imshow('result', result)

# result = weighted_img(hough_img, image)

# cv2.imshow('canny', canny_img)
# cv2.imshow('roi result', roi_img)
# cv2.imshow('hough result', hough_img)
# cv2.imshow('houghP result', hough_img)
# cv2.imshow('result', result)

cv2.waitKey(0)