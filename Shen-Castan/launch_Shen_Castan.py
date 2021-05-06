import numpy as np
import ShenCastanFlat as filter
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

img = 'Experiment-118-cut4.tiff'
filtering_engine = filter.ShenCastanFlat(img)

filtered_img = filtering_engine.filter(alpha=0.8, flat_radius=4, filter_radius=64)
gaussian_img = gaussian_filter(filtered_img, sigma=16)
corrected_img = filtered_img/gaussian_img
binary_img = np.where(corrected_img <0.8, 1, 0)

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
ax3.title.set_text('Filtered image - blurred')
ax3.axis('off')
ax3.imshow(corrected_img)

ax4 = fig.add_subplot(2, 2, 4)
ax4.title.set_text('Mask image')
ax4.axis('off')
ax4.imshow(binary_img)

plt.show()

