import math

import numpy as np
import ShenCastanFlat as filter
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
import cv2

img = 'Experiment-118-cut4.tiff'
filtering_engine = filter.ShenCastanFlat(img)

filtered_img = filtering_engine.filter(alpha=0.8, flat_radius=4, filter_radius=64)
gaussian_img = gaussian_filter(filtered_img, sigma=16)
corrected_img = filtered_img / gaussian_img
binary_img = np.where(corrected_img < 0.8, 1, 0)

fig = plt.figure(figsize=(8.25, 11.75))

ax1 = fig.add_subplot(2, 2, 1)
ax1.title.set_text('Original image')
ax1.axis('off')
ax1.imshow(filtering_engine.input_image)

ax2 = fig.add_subplot(2, 2, 2)
ax2.title.set_text('Filtered image')
ax2.axis('off')
ax2.imshow(filtered_img)

ax3 = fig.add_subplot(2, 2, 3)
ax3.title.set_text('Filtered - blurred image')
ax3.axis('off')
ax3.imshow(corrected_img)

ax4 = fig.add_subplot(2, 2, 4)
ax4.title.set_text('Mask image')
ax4.axis('off')
ax4.imshow(binary_img)

plt.show()

mask = cv2.normalize(src=binary_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ori = cv2.normalize(src=filtering_engine.input_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2RGB)

cv2.imshow("Mask", mask)
cv2.waitKey(0)

lines = cv2.HoughLines(mask, 1, np.pi / 180, 150, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
#         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
#         cv2.line(ori, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
#         print(i)

linesP = cv2.HoughLinesP(mask, 1, np.pi / 180, 64, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(ori, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow("Detections", ori)
cv2.waitKey(0)
