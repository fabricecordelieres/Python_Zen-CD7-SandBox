# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
import math as math

img_path = "Experiment-118-cut4.tiff"

# load image as pixel array
image = image.imread(img_path)

image = np.dot(image[..., :3], [0.299, 0.587, 0.114])  # Convert RGB to Grays

# summarize shape of the pixel array
print(image.dtype)
print(image.shape)

# Formules issues de  http://devernay.free.fr/cours/vision/pdf/c3.pdf
filter_radius = 24
alpha = 0.35

# ------------------------ LISSAGE ------------------------
kernel_liss = np.zeros((filter_radius * 2 + 1))
c = (1 - math.exp(-alpha)) / (1 + math.exp(-alpha))
kernel_fct = (lambda c_param, alpha_param, x_param: c_param * math.exp(-alpha_param * math.fabs(x_param)))
for x in range(-filter_radius, filter_radius + 1, 1):
    kernel_liss[x + filter_radius] = kernel_fct(c, alpha, x)
print(kernel_liss)

# Convolve x only
convolve_x = []
for i in range(0, image.shape[0], 1):
    convolve_x.append(np.convolve(image[i, :], kernel_liss, mode='same'))

convolve_x = np.array(convolve_x)
convolve_x = np.transpose(convolve_x)

convolve_xy = []
for i in range(0, convolve_x.shape[0], 1):
    convolve_xy.append(np.convolve(convolve_x[i, :], kernel_liss, mode='same'))

convolve_xy = np.array(convolve_xy)
convolve_xy = np.transpose(convolve_xy)


# ------------------------ DERIVATIF ------------------------
kernel_der = np.zeros((filter_radius * 2 + 1))
d = 1 - math.exp(-alpha)
kernel_fct = (lambda d_param, alpha_param, x_param: d_param * math.exp(-alpha_param * np.fabs(x_param)) if x_param>=0 else -d_param * math.exp(-alpha_param * np.fabs(x_param)))
for x in range(-filter_radius, filter_radius + 1, 1):
    kernel_der[x + filter_radius] = kernel_fct(d, alpha, x)

print(kernel_der)

# Convolve x only
convolve_der_x = []
for i in range(0, image.shape[0], 1):
    convolve_der_x.append(np.convolve(convolve_xy[i, :], kernel_der, mode='same'))

convolve_der_x = np.array(convolve_der_x)
convolve_der_x = np.transpose(convolve_der_x)

convolve_der_xy = []
for i in range(0, convolve_der_x.shape[0], 1):
    convolve_der_xy.append(np.convolve(convolve_der_x[i, :], kernel_der, mode='same'))

convolve_der_xy = np.array(convolve_der_xy)
convolve_der_xy = np.transpose(convolve_der_xy)

delta_xy = convolve_der_xy - image

fig = plt.figure()

ax1 = fig.add_subplot(3, 2, 1)
ax1.title.set_text('Original image')
ax1.axis('off')
ax1.imshow(image)
ax2 = fig.add_subplot(3, 2, 2)
ax2.title.set_text('Smooth, a='+str(alpha)+" rad="+str(filter_radius))
ax2.axis('off')
ax2.imshow(convolve_xy)
ax3 = fig.add_subplot(3, 2, 3)
ax3.title.set_text('Derivative')
ax3.axis('off')
ax3.imshow(convolve_der_xy)
ax4 = fig.add_subplot(3, 2, 4)
ax4.title.set_text('Diff (deriv-img)/img')
ax4.axis('off')
ax4.imshow(delta_xy)

ax5 = fig.add_subplot(3, 2, 5)
ax5.title.set_text('Kernel lissage')
ax5.plot(range(-filter_radius, filter_radius + 1, 1), kernel_liss)

ax6 = fig.add_subplot(3, 2, 6)
ax6.title.set_text('Kernel derivatif')
ax6.plot(range(-filter_radius, filter_radius + 1, 1), kernel_der)

plt.show()
