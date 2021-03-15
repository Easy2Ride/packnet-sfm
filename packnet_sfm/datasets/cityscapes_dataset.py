
import re
from collections import defaultdict
import os

from torch.utils.data import Dataset
import numpy as np
from packnet_sfm.utils.image import load_image

from packnet_sfm.datasets.augmentations import resize_sample, random_center_crop_sample

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

Kzero=np.array([[655.4526442798641, 0.0             , 939.3564129071235],
                [0.0              , 655.229038531984, 536.3086826317193],
                [0.0              , 0.0             , 1.0              ]])

Kacquisition=np.array([[407.5891, 0.0     , 648.9632],
                       [0.0     , 407.7303, 360.3419],
                       [0.0     , 0.0     , 1.0     ]])

def dummy_calibration(image):
    w, h = [float(d) for d in image.size]
    return np.array([[1000. , 0.    , w / 2. - 0.5],
                     [0.    , 1000. , h / 2. - 0.5],
                     [0.    , 0.    , 1.          ]])
                     

def cityscapes_calibration(image):
    w, h = [float(d) for d in image.size]
    K = np.array([[1.104, 0, 0.5],
                 [0, 2.212, 0.5],
                 [0, 0, 1]], dtype=np.float32) 
    K[0,0] *= w
    K[0,2] *= w
    K[1,1] *= h 
    K[1,2] *= h
    return K

def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

def read_files(directory, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            d_files = read_files(entry.path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath] = d_files[entry.path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                files[directory].append(relpath)
    return files

########################################################################################################################
#### DATASET
########################################################################################################################

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, file_list, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type=None, **kwargs):
        super().__init__()
        # Asserts
        assert depth_type is None or depth_type == '', \
            'CityscapesDataset currently does not support depth types'
        assert len(strides) == 1 and strides[0] == 1, \
            'CityscapesDataset currently only supports stride of 1.'

        self.root_dir = root_dir
        self.file_list = file_list

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = 1

        self.files = []
        file_tree = read_files(root_dir)
        
        with open(file_list, "r") as f:
            data = f.readlines()

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data):
            path = os.path.join(root_dir, fname.split()[0])
            self.paths.append(path)
        
        #for k, v in file_tree.items():
        #    file_set = set(file_tree[k])
        #    files = [fname for fname in sorted(v) if self._has_context(fname, file_set)]
        #    self.files.extend([[k, fname] for fname in files])

        self.data_transform = data_transform

    def __len__(self):
        return len(self.paths)

    def _change_idx(self, idx, filename):
        #_, ext = os.path.splitext(os.path.basename(filename))
        #return self.split.format(idx) + ext
        base, ext = os.path.splitext(os.path.basename(filename))
        parts = base.split('_')
        parts[-2] = str(idx).zfill(len(parts[-2]))
        base = '_'.join(parts)
        return os.path.join(os.path.dirname(filename), base + ext)

    def _has_context(self, filename, file_set):
        context_paths = self._get_context_file_paths(filename)
        return all([f in file_set for f in context_paths])

    def _get_context_file_paths(self, filename):
        base, ext = os.path.splitext(os.path.basename(filename))
        f_idx = int(base.split('_')[-2]) # get_idx(filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
               list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        paths = [self._change_idx(f_idx + i, filename) for i in idxs]
        return [fname for fname in paths if os.path.exists(fname)]

    def _read_rgb_context_files(self, filename):
        context_paths = self._get_context_file_paths(filename)
        
        return [load_image(os.path.join(self.root_dir, filename))
                for filename in context_paths]

    def _read_rgb_file(self, filename):
        return load_image(os.path.join(self.root_dir, filename))

    def __getitem__(self, idx):
        filename = self.paths[idx]
        image = self._read_rgb_file(filename)

        # intrinsics
        if "zero" in self.root_dir.lower():
            intrinsics=Kzero
        elif "acquisition" in self.root_dir.lower():
            intrinsics = Kacquisition
        else:
            intrinsics = dummy_calibration(image)
            
        intrinsics = cityscapes_calibration(image)

        session = self.file_list.split('/')[-1].split('.')[0]
        # dict to return
        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session, os.path.splitext(filename)[0]),
            #
            'pose': np.ones((4,4))*np.nan, # placeholder
            'rgb': image,
            'intrinsics': intrinsics
        }

        if self.has_context:
            sample['rgb_context'] = \
                self._read_rgb_context_files(filename)

        # Placeholder for pose context
        sample['pose_context'] = [np.ones((4,4))*np.nan for i in sample['rgb_context']]

        # Resize sample perserving aspect ratio
        sample=resize_sample(sample, shape=(320,640))
        sample=random_center_crop_sample(sample, size=(192,640))

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
