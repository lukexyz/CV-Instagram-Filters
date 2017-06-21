import skimage
from skimage import io, filters
# from skimage.viewer import ImageViewer
import numpy as np

print('Instagram Filter Remake: Gotham')
original_image = skimage.io.imread('images/zell_am_see_snowboarding.jpg')
original_image = skimage.util.img_as_float(original_image)


def split_image_into_channels(image):
    """Look at each image separately"""
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel


def merge_channels(red, green, blue):
    """Merge channels back into an image"""
    return np.stack([red, green, blue], axis=2)

r, g, b = split_image_into_channels(original_image)
im = merge_channels(r, g, b)


def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = skimage.filters.gaussian_filter(image, sigma=10, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 1.0)
    return sharper


def channel_adjust(channel, values):
    # preserve the original size, so we can reconstruct at the end
    orig_size = channel.shape
    # flatten the image into a single array
    flat_channel = channel.flatten()

    # this magical numpy function takes the values in flat_channel
    # and maps it from its range in [0, 1] to its new squeezed and
    # stretched range
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)

    # put back into the original image shape
    return adjusted.reshape(orig_size)

# 1. Colour channel adjustment example
r, g, b = split_image_into_channels(original_image)
r_interp = channel_adjust(r, [0, 0.8, 1.0])
red_channel_adj = merge_channels(r_interp, g, b)
skimage.io.imsave('images/1_red_channel_adj.jpg', red_channel_adj)

# 2. Mid tone colour boost
r, g, b = split_image_into_channels(original_image)
r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
r_boost_img = merge_channels(r_boost_lower, g, b)
skimage.io.imsave('images/2_mid_tone_colour_boost.jpg', r_boost_img)

# 3. Making the blacks bluer
bluer_blacks = merge_channels(r_boost_lower, g, np.clip(b + 0.03, 0, 1.0))
skimage.io.imsave('images/3_bluer_blacks.jpg', bluer_blacks)

# 4. Sharpening the image
sharper = sharpen(bluer_blacks, 1.3, 0.3)
skimage.io.imsave('images/4_sharpened.jpg', sharper)

# 5. Blue channel boost in lower-mids, decrease in upper-mids
r, g, b = split_image_into_channels(sharper)
b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
gotham = merge_channels(r, g, b_adjusted)
skimage.io.imsave('images/5_blue_adjusted.jpg', gotham)
