# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from packnet_sfm.geometry.camera_utils import scale_intrinsics

from PIL import Image

from packnet_sfm.utils.misc import filter_dict

########################################################################################################################

def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)

def resize_sample_image_and_intrinsics(sample, shape,
                                       image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample['rgb'].size
    (out_h, out_w) = shape
    # Scale intrinsics
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        sample[key] = scale_intrinsics(np.copy(sample[key]),
                                        out_w / orig_w, 
                                        out_h / orig_h)
    # Scale images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original',
    ]):
        sample[key] = image_transform(sample[key])
    # Scale context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original',
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]
    # Return resized sample
    return sample

def resize_sample(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)
    # Resize depth maps
    for key in filter_dict(sample, [
        'depth',
    ]):
        sample[key] = resize_depth(sample[key], shape)
    # Resize depth contexts
    for key in filter_dict(sample, [
        'depth_context',
    ]):
        sample[key] = [resize_depth(k, shape) for k in sample[key]]
    # Return resized sample
    return sample

########################################################################################################################

def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    # Return converted sample
    return sample

########################################################################################################################

def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare transformation
        color_augmentation = transforms.ColorJitter()
        brightness, contrast, saturation, hue = parameters
        augment_image = color_augmentation.get_params(
            brightness=[max(0, 1 - brightness), 1 + brightness],
            contrast=[max(0, 1 - contrast), 1 + contrast],
            saturation=[max(0, 1 - saturation), 1 + saturation],
            hue=[-hue, hue])
        # Jitter single items
        for key in filter_dict(sample, [
            'rgb'
        ]):
            sample[key] = augment_image(sample[key])
        # Jitter lists
        for key in filter_dict(sample, [
            'rgb_context'
        ]):
            sample[key] = [augment_image(k) for k in sample[key]]
    # Return jittered (?) sample
    return sample

def rotate_sample(sample, degrees=20):
    """
    Rotates input images as data augmentation.
    Assumes the intrinsics to be scaled w.r.t the image. i.e. this step should be followed by (image &intrinsics) resizing

    Parameters
    ----------
    sample : dict
        Input sample
    degrees : float
        sample a random rotation angle between [-degrees,, degrees]

    Returns
    -------
    sample : dict
        Rotated sample

    Author: Zeeshan Khan Suri
    """    
    if degrees == 0:
        return sample
    # Do same rotation transform for image and context imgs
    rand_degree = (torch.rand(1)-0.5)*2*degrees
    # Get rotation center as principal point from intrinsic matrix
    center=sample["intrinsics"][:2,2]
    # Jitter single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth'
    ]):
        sample[key] = TF.rotate(sample[key], rand_degree.item(),
                                resample=Image.NEAREST if key=='depth' else Image.BILINEAR,
                                center=center.tolist())
    # Jitter lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [TF.rotate(k,rand_degree.item(),
                                 resample=Image.NEAREST if key=='depth_context' else Image.BILINEAR,
                                 center=center.tolist()) 
                for k in sample[key]]
    # Return rotated (?) sample
    return sample


def random_center_crop_sample(sample, size=None):
    """
    Random center crops of sample as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    size : (sequence of (h,w))
        Desired output size of the crop. (h, w)

    Returns
    -------
    sample : dict
        Cropped sample
    """
    
    (w, h) = sample['rgb'].size
    if size:
        crop_h = list(range(64,h+1,32)) # Since network needs multiples of 32
        crop_h = np.random.choice(crop_h,p=crop_h/np.sum(crop_h))
        crop_w = list(range(64,w+1,32))
        crop_w = np.random.choice(crop_w,p=crop_w/np.sum(crop_w))
    else:
        crop_h, crop_w = size
    center_crop = transforms.CenterCrop((crop_h, crop_w))

    # Center cropping does **not** change (normalized) image intrinsics but position changes
    sample["intrinsics"][0,2] = sample["intrinsics"][0,2]*(crop_w/w)
    sample["intrinsics"][1,2] = sample["intrinsics"][1,2]*(crop_h/h)

    # So, only cropping the rest
    for key in filter_dict(sample, [
        'rgb', 
    ]):
        sample[key] = center_crop(sample[key])
    # Jitter lists
    for key in filter_dict(sample, [
        'rgb_context',
    ]):
        sample[key] = [center_crop(k) for k in sample[key]]
    # Return random cropped (?) sample
    return sample

########################################################################################################################
