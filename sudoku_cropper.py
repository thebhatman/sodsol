import cv2
import math
import numpy as np

img = cv2.imread("sudoku-original.jpg")
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#blur_img = cv2.GaussianBlur(img, (11,11), 0)
img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5), np.uint8)
#cv2.imshow("image", blur_img)
thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh_img = cv2.bitwise_not(thresh_img, thresh_img)
# kernel = np.ones((5,5), np.uint8)
# dilated_img = cv2.dilate(thresh_img, kernel, iterations = 5)
edges_img = cv2.Canny(thresh_img, 50, 150, apertureSize = 3)
im,contour_list, heirarchy = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_area = 0
largest_contour_index = -1
for i in range(len(contour_list)-3):
	area = cv2.contourArea(contour_list[i])
	if area > largest_area:
		largest_area = area
		largest_contour_index = i

print(img.shape)
new_img = np.zeros(img.shape)
cv2.drawContours(new_img, contour_list, largest_contour_index, (255,255,255), 2)
sudoku_img = np.zeros_like(thresh_img)
#print(contour_list[largest_contour_index][1])
# minline_length = 28
# maxline_gap = 30
# lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, minline_length, maxline_gap)
# for i in range(len(lines)):
# 	print(lines[i])
# 	for x1, y1, x2, y2 in lines[i]:
# 		cv2.line(new_img, (x1,y1),(x2,y2), (0,255,0),2)
for i in range(len(contour_list)):
	approx_shape = cv2.approxPolyDP(contour_list[i], 0.01*cv2.arcLength(contour_list[i],True), True)
	if(len(approx_shape) == 4):
		cv2.drawContours(new_img, [contour_list[i]], 0, (255,255,255), 2)
# parameters = cv2.SimpleBlobDetector_Params()
# parameters.minThreshold = 10
# parameters.maxThreshold = 200
# parameters.filterByArea = False
# parameters.minArea = 100
# parameters.filterByCircularity = True
# parameters.minCircularity = 0.785
# parameters.filterByInertia = False
# square_detector = cv2.SimpleBlobDetector_create(parameters)
# keypoints = square_detector.detect(thresh_img)
# blob_img = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

new_img = cv2.dilate(new_img, kernel, iterations = 1)
new_img = cv2.erode(new_img, kernel, iterations = 2)
# #new_img = cv2.GaussianBlur(new_img, (3,3), 10)
# minline_length = 200
# maxline_gap = 30
# lines = cv2.HoughLines(edges_img, 1, np.pi/180, minline_length)
# print(lines)
# for r,theta in lines[0]: 
#     a = np.cos(theta)  
#     b = np.sin(theta)  
#     x0 = a*r  
#     y0 = b*r  
#     x1 = int(x0 + 1000*(-b))  
#     y1 = int(y0 + 1000*(a))  
#     x2 = int(x0 - 1000*(-b))  
#     y2 = int(y0 - 1000*(a))   
#     cv2.line(new_img,(x1,y1), (x2,y2), (0,0,255),2) 

#for contour in contour_list:
#	if (cv2.contourArea(contour) > cv2.contourArea(contour_list[largest_contour_index])/81 - 1*cv2.contourArea(contour)) and (cv2.contourArea(contour) < cv2.contourArea(contour_list[largest_contour_index])/81 + 1*cv2.contourArea(contour)):
#		cv2.drawContours(new_img, [contour], 0, (255,255,255), 2)
x, y, w, h = cv2.boundingRect(contour_list[largest_contour_index])
cropped_img = img[y:y + h, x:x + w]

def get_rectangle_corners(cnt):
	#print(cnt)
	pts = cnt.reshape(4, 2)  ## cnt stores the vertices of the sudoku rectangle obtained by approxPoly
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def warp_perspective(rect, grid):     ##Function makes the sudoku grid planar!
	(tl, tr, br, bl) = rect     ## Topright, topleft, bottomright, bottomleft
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warp = cv2.warpPerspective(grid, M, (maxWidth, maxHeight))  ##Magical Opencv function to warp the image
	return warp


warp_img = warp_perspective(get_rectangle_corners(cv2.approxPolyDP(contour_list[largest_contour_index], 0.01*cv2.arcLength(contour_list[largest_contour_index],True), True)), gray_img)
print(warp_img.shape)
warp_img = cv2.adaptiveThreshold(warp_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#warp_img = cv2.GaussianBlur(warp_img, (9,9), 0)
warp_resize_img = cv2.resize(warp_img, (252,252))

cell = []
for i in range(81):
	cell.append(np.zeros((28,28)))

for i in range(9):
	for j in range(9):
		cell[i*9 + j%9] = warp_resize_img[i*28:(i+1)*28, j*28:(j+1)*28]

for i in range(81):
	cv2.imshow("image", cell[i])
	cv2.waitKey(1000)