import os
import cv2
import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, path, input_size):
        self.input_size = input_size
        self.path = path
        labels = self.load_labels(path)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())
        self.imagenames = [os.path.join(path, 'images', 'train', f'{os.path.basename(i)[:-4]}.png') for i in self.filenames]
        self.n = len(self.filenames)
        self.indices = range(self.n)

    def __getitem__(self, index):
        index = self.indices[index]
        image = self.load_image(index)
        h, w = image.shape
        image, ratio, pad = resize(image, self.input_size)
        label = self.labels[index].copy()
        if label.size:
            label[:, 1:] = wh2xy(label[:, 1:], ratio * w, ratio * h, pad[0], pad[1])

        nl = len(label)
        h, w = image.shape
        clas = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(clas)
            target_box = torch.from_numpy(box)

        sample = image.reshape((1, h, w))
        sample = np.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.imagenames[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def collate_fn(batch):
        samples, clas, box, indices = zip(*batch)

        clas = torch.cat(clas, dim=0)
        box = torch.cat(box, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'cls': clas,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets

    @staticmethod
    def load_labels(path):
        saved = os.path.join(path, 'x.cache')
        if os.path.exists(saved):
            return torch.load(saved)
        path = os.path.join(path, 'labels', 'train')
        filenames = [os.path.join(path, i) for i in os.listdir(path)]
        x = {}
        for filename in filenames:
            with open(filename) as f:
                label = [i.split() for i in f.read().strip().splitlines() if len(i)]
                label = np.array(label, dtype=np.float32)
            nl = len(label)
            if not nl:
                label = np.zeros((0, 5), dtype=np.float32)
            x[filename] = label
        torch.save(x, saved)
        return x


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def resize(image, input_size):
    shape = image.shape[:2]
    r = min(input_size / shape[0], input_size / shape[1])
    r = min(r, 1.0)

    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:
        image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image, r, (w, h)
