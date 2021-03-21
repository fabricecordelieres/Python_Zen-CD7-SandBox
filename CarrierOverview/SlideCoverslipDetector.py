import cv2  # pip install opencv-python
import numpy as np


class SlideCoverslipDetector:
    """
    This class is aimed at finding the slides and coverslips from a CarrierOverview object
    """

    coo = ''

    def __init__(self, carrier_overview=''):
        """
        Constructor: creates a new SlideCoverslipDetector object
        :param carrier_overview: the CarrierOverview object to work on
        """

        self.coo = carrier_overview

    def test_detect(self):
        img = self.coo.get_image()
        img_out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
        bilateral_filter = cv2.bilateralFilter(img, 17, 9, 36)
        adaptive_thr = cv2.adaptiveThreshold(bilateral_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                             255, 0)
        #edges = cv2.Canny(adaptive_thr, 0, 0)

        # -----------------------------------------------------
        circles = cv2.HoughCircles(adaptive_thr, cv2.HOUGH_GRADIENT, 2, img.shape[1] / 10,
                                   param1=100, param2=65,
                                   minRadius=150, maxRadius=200)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(img_out, center, 3, (255, 0, 0), 5)
                # circle outline
                radius = i[2]
                cv2.circle(img_out, center, radius, (255, 0, 0), 5)
                print(i)
        # -----------------------------------------------------
        slides, hierarchy = cv2.findContours(adaptive_thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Only want the contours, not the hierarchy

        for slide in slides:
            bbox = cv2.boundingRect(slide)
            x, y, w, h = bbox  # unpacking
            if 800 < w < 1000 and 1800 < h < 2000 :
                cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 5)
                print(bbox)

        zoom = 0.15
        w = int(img.shape[1] * zoom)
        h = int(img.shape[0] * zoom)
        offset_x = 0
        offset_y = 0

        cv2.imshow('Ori', img)
        cv2.imshow('Bilateral_filter', bilateral_filter)
        cv2.imshow('Adaptive_threshold', adaptive_thr)
        #cv2.imshow('Edges', edges)
        cv2.imshow('Detections', img_out)

        cv2.namedWindow('Ori', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Bilateral_filter', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Adaptive_threshold', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('Ori', w, h)
        cv2.resizeWindow('Bilateral_filter', w, h)
        cv2.resizeWindow('Adaptive_threshold', w, h)
        #cv2.resizeWindow('Edges', w, h)
        cv2.resizeWindow('Detections', w, h)

        offset_x = offset_x + w
        cv2.moveWindow('Bilateral_filter', offset_x, offset_y)
        offset_x = offset_x + w
        cv2.moveWindow('Adaptive_threshold', offset_x, offset_y)
        offset_x = offset_x + w
        #cv2.moveWindow('Edges', offset_x, offset_y)
        #offset_x = 0
        #offset_y = offset_y + h
        cv2.moveWindow('Detections', offset_x, offset_y)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
