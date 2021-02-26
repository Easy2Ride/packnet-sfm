# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import partial
from torchvision import transforms
from packnet_sfm.datasets.augmentations import resize_image, resize_sample, \
    duplicate_sample, colorjitter_sample, to_tensor_sample, rotate_sample, \
    random_center_crop_sample

########################################################################################################################

def train_transforms(sample, image_shape, jittering, **kwargs):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters

    Returns
    -------
    sample : dict
        Augmented sample
    """
    # (orig_w, orig_h) = sample['rgb'].size
    # (h, w) = shape
    # factor = w/orig_w
    # h=h*factor

    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)
    if "max_roll_angle" in kwargs:
        sample = rotate_sample(sample, degrees=kwargs["max_roll_angle"])
    if "random_center_crop" in kwargs:
        if kwargs["random_center_crop"]:
            sample = random_center_crop_sample(sample)
    sample = to_tensor_sample(sample)
    return sample

def validation_transforms(sample, image_shape):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)
    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape, jittering, **kwargs):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    max_roll_angle : float
        Rotation augmentation with a random angle between [-max_roll_angle, max_roll_angle]
    random_center_crop : bool
        To do random center crop or not

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       **kwargs)
    elif mode == 'validation':
        return partial(validation_transforms,
                       image_shape=image_shape)
    elif mode == 'test':
        return partial(test_transforms,
                       image_shape=image_shape)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

