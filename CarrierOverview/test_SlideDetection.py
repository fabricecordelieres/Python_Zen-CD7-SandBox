from CarrierOverview import CarrierOverview as co
from ContainerCoverslipDetector import ContainerCoverslipDetector as cc
import cv2
import numpy as np

input_dir = './datasets/'
filename = 'CarrierOverview.czi'
co_obj = co(input_dir, filename)
cc_obj = cc(co_obj)
containers=cc_obj.find_containers() # Options are available: width, height and tolerance
x, y, w, h= containers['Container1']['Coordinates']


img = cv2.cvtColor(cc_obj.get_image()[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)

#-------Test Gamma
gamma=0.5
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
img = cv2.LUT(img, lookUpTable)



bilateral_filter = cv2.bilateralFilter(img, 17, 9, 36)
adaptive_thr = cv2.adaptiveThreshold(bilateral_filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 129, 0)
slides, hierarchy = cv2.findContours(adaptive_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Only want the contours, not the hierarchy

img_out=cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
for slide in slides:
    bbox = cv2.boundingRect(slide)
    x, y, w, h = bbox  # unpacking
    if 500 < w < 800 and 1500 < h < 1800:
        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 5)



zoom = 0.15
w = int(img.shape[1] * zoom)
h = int(img.shape[0] * zoom)
offset_x = 0
offset_y = 0
cv2.imshow('Ori', img)
cv2.imshow('Bilateral_filter', bilateral_filter)
cv2.imshow('Adaptive_threshold', adaptive_thr)
# cv2.imshow('Edges', edges)
cv2.imshow('Detections', img_out)

cv2.namedWindow('Ori', cv2.WINDOW_NORMAL)
cv2.namedWindow('Bilateral_filter', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adaptive_threshold', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Ori', w, h)
cv2.resizeWindow('Bilateral_filter', w, h)
cv2.resizeWindow('Adaptive_threshold', w, h)
# cv2.resizeWindow('Edges', w, h)
cv2.resizeWindow('Detections', w, h)

offset_x = offset_x + w
cv2.moveWindow('Bilateral_filter', offset_x, offset_y)
offset_x = offset_x + w
cv2.moveWindow('Adaptive_threshold', offset_x, offset_y)
offset_x = offset_x + w
# cv2.moveWindow('Edges', offset_x, offset_y)
# offset_x = 0
# offset_y = offset_y + h
cv2.moveWindow('Detections', offset_x, offset_y)

cv2.waitKey(0)
cv2.destroyAllWindows()


#print(cc_obj.find_circular_coverslips(limit_to_area=(250, 0, 1250, 2030))) # Options are available: diameter and tolerance
#cc_obj.find_circular_coverslips(diameter= 20, tolerance=.2)
#cc_obj.find_circular_coverslips_in_containers(diameter_coverslip= 20, tolerance_coverslip=.1)

#cc_obj.find_circular_coverslips_in_containers()

#cc_obj.show()

