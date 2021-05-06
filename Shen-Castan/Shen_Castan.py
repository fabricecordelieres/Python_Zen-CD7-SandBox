# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


# ------------------------ 2D CONVOLUTION ------------------------
def convolve_2d(img, kernel, dir='xy'):
    """
    Performs a 2D convolution based on an input 2D numpy matrix and a 1D kernel
    Parameters:
        img (2D numpy array): the image to convolve
        kernel (array): the kernel
        dir (string): the direction(s) of convolution (might be x, y or xy)
    Return:
        the convolved image, as a numpy array
    """

    convolved_x = []

    # Convolve along x axis
    if dir.find('x') != -1:
        for i in range(0, image.shape[0], 1):
            convolved_x.append(np.convolve(img[i, :], kernel, mode='same'))

        convolved = np.array(convolved_x)

    # Convolve along y axis
    if dir.find('y') != -1:
        convolved_y = []
        # Initialize convolved in cas only y convolution has been called
        if dir.find('x') == -1:
            convolved = img.copy()

        convolved = np.transpose(convolved)

        for i in range(0, convolved.shape[0], 1):
            convolved_y.append(np.convolve(convolved[i, :], kernel, mode='same'))

        convolved = np.array(convolved_y)
        convolved = np.transpose(convolved)

    return convolved


# ------------------------ 2D CONVOLUTION ------------------------
img_path = "Experiment-118-cut4.tiff"

# load image as pixel array
image = image.imread(img_path)
image_original = image.copy();

image = np.dot(image[..., :3], [0.299, 0.587, 0.114])  # Convert RGB to Grays

#image = np.log(image+0.000000001)

# Formulas from http://devernay.free.fr/cours/vision/pdf/c3.pdf
filter_radius = 64
alpha = 0.25

# ------------------------ SMOOTHING ------------------------
kernel_smooth = np.zeros((filter_radius * 2 + 1))
c = (1 - np.exp(-alpha)) / (1 + np.exp(-alpha))
kernel_fct = (lambda c_param, alpha_param, x_param: c_param * np.exp(-alpha_param * np.fabs(x_param)))
for x in range(-filter_radius, filter_radius + 1, 1):
    kernel_smooth[x + filter_radius] = kernel_fct(c, alpha, x)

convolve_smooth = convolve_2d(image, kernel_smooth, 'xy')

# ------------------------ DERIVATIVE ------------------------
kernel_der = np.zeros((filter_radius * 2 + 1))
d = 1 - np.exp(-alpha)
kernel_fct = (lambda d_param, alpha_param, x_param: d_param * np.exp(
    -alpha_param * np.fabs(x_param)) if x_param >= 0 else -d_param * np.exp(-alpha_param * np.fabs(x_param)))
for x in range(-filter_radius, filter_radius + 1, 1):
    kernel_der[x + filter_radius] = kernel_fct(d, alpha, x)

convolve_der_x = convolve_2d(convolve_smooth, kernel_der, 'x')
convolve_der_y = convolve_2d(convolve_smooth, kernel_der, 'y')

convolve_der_xy_amplitude = np.sqrt(np.square(convolve_der_x) + np.square(convolve_der_y))

convolve_der_xy_orientation = np.arctan2(convolve_der_y, convolve_der_x)

# ------------------------ FIGURE ------------------------
fig = plt.figure(figsize=(8.25, 11.75))

ax1 = fig.add_subplot(4, 2, 1)
ax1.title.set_text('Original image')
ax1.axis('off')
ax1.imshow(image_original)

ax2 = fig.add_subplot(4, 2, 2)
ax2.title.set_text('Smooth, a=' + str(alpha) + " rad=" + str(filter_radius))
ax2.axis('off')
ax2.imshow(convolve_smooth)

ax3 = fig.add_subplot(4, 2, 3)
ax3.title.set_text('Derivative X')
ax3.axis('off')
ax3.imshow(convolve_der_x)

ax4 = fig.add_subplot(4, 2, 4)
ax4.title.set_text('Derivative Y')
ax4.axis('off')
ax4.imshow(convolve_der_y)

ax5 = fig.add_subplot(4, 2, 5)
ax5.title.set_text('Derivative XY\nAmplitude')
ax5.axis('off')
ax5.imshow(convolve_der_xy_amplitude)

ax5 = fig.add_subplot(4, 2, 6)
ax5.title.set_text('Derivative XY\nOrientation')
ax5.axis('off')
ax5.imshow(convolve_der_xy_orientation)

ax7 = fig.add_subplot(4, 2, 7)
ax7.title.set_text('Kernel smoothing')
ax7.plot(range(-filter_radius, filter_radius + 1, 1), kernel_smooth)

ax8 = fig.add_subplot(4, 2, 8)
ax8.title.set_text('Kernel derivative')
ax8.plot(range(-filter_radius, filter_radius + 1, 1), kernel_der)

plt.show()

im = Image.fromarray(convolve_der_xy_amplitude)
im.save('export_a=' + str(alpha) + '_rad=' + str(filter_radius) + '.tif')
