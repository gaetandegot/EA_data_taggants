import numpy as np

import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import (
    affine, rotate, autocontrast, equalize, invert, solarize, posterize,
    adjust_contrast, adjust_saturation, adjust_brightness, adjust_sharpness,
    gaussian_blur
)

_LEVEL_DENOM = 10.
# denominator for conversion from 'Mx' magnitude scale to fractional aug level

_DEFAULT_INTERPOLATION = torchvision.transforms.InterpolationMode.BILINEAR


def shear_x(img, factor, **kwargs):
    sx = np.arctan(factor) * 180 / np.pi
    tx = - img.shape[0] * factor / 2
    return affine(img, angle=0, translate=[tx, 0], scale=1.0, shear=[sx, 0],
                  interpolation=_DEFAULT_INTERPOLATION,
                  fill=kwargs.get('fillcolor', 0))


def shear_y(img, factor, **kwargs):
    sy = np.arctan(factor) * 180 / np.pi
    ty = - img.shape[1] * factor / 2
    return affine(img, angle=0, translate=[0, ty], scale=1.0, shear=[0, sy],
                  interpolation=_DEFAULT_INTERPOLATION,
                  fill=kwargs.get('fillcolor', 0))


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.shape[0]
    return affine(img, angle=0, translate=[pixels, 0], scale=1.0, shear=[0, 0],
                  interpolation=_DEFAULT_INTERPOLATION,
                  fill=kwargs.get('fillcolor', 0))


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.shape[1]
    return affine(img, angle=0, translate=[0, pixels], scale=1.0, shear=[0, 0],
                  interpolation=_DEFAULT_INTERPOLATION,
                  fill=kwargs.get('fillcolor', 0))


def translate_x_abs(img, pixels, **kwargs):
    return affine(img, angle=0, translate=[pixels, 0], scale=1.0, shear=[0, 0],
                  interpolation=_DEFAULT_INTERPOLATION,
                  fill=kwargs.get('fillcolor', 0))


def translate_y_abs(img, pixels, **kwargs):
    return affine(img, angle=0, translate=[0, pixels], scale=1.0, shear=[0, 0],
                  interpolation=_DEFAULT_INTERPOLATION,
                  fill=kwargs.get('fillcolor', 0))


def rotate_(img, degrees, **kwargs):
    return rotate(img, degrees, interpolation=_DEFAULT_INTERPOLATION)


def auto_contrast(img, **__):
    return autocontrast(img)


def invert_(img, **__):
    return invert(img)


def equalize_(img, **__):
    return img + (equalize((255*img).byte())/255 - img).detach()


def solarize_(img, thresh, **__):
    return solarize(img, thresh/255)


def solarize_add(img, add, thresh=128, **__):
    return img + add/255 * (img < thresh/255)


def posterize_(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return img + (posterize((255*img).byte(), bits_to_keep)/255 - img).detach()


def contrast(img, factor, **__):
    return adjust_contrast(img, factor)


def color(img, factor, **__):
    return adjust_saturation(img, factor)


def brightness(img, factor, **__):
    return adjust_brightness(img, factor)


def sharpness(img, factor, **__):
    return adjust_sharpness(img, factor)


def gaussian_blur_(img, factor, **__):
    return gaussian_blur(img, kernel_size=21, sigma=factor)


def gaussian_blur_rand(img, factor, **__):
    radius_min = 0.1
    radius_max = 2.0
    return transforms.GaussianBlur(
        kernel_size=21,
        sigma=(radius_min, radius_max * factor)
    )(img)


def desaturate(img, factor, **_):
    factor = min(1., max(0., 1. - factor))
    # enhance factor 0 = grayscale, 1.0 = no-change
    return color(img, factor, **_)


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize_,
    'Invert': invert_,
    'Rotate': rotate_,
    'Posterize': posterize_,
    'PosterizeIncreasing': posterize_,
    'PosterizeOriginal': posterize_,
    'Solarize': solarize_,
    'SolarizeIncreasing': solarize_,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
    'Desaturate': desaturate,
    'GaussianBlur': gaussian_blur_,
    'GaussianBlurRand': gaussian_blur_rand,
}
