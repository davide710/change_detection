import torch
import numpy as np
import cv2


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

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    r = 640 / max(h, w)
    if r != 1:
        image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    image = 1 - image / 255.0
    image[image > 0] = 1
    image, ratio, pad = resize(image, 640)
    h, w = image.shape

    sample = image.reshape((1, 1, h, w))
    sample = np.ascontiguousarray(sample)
    sample = torch.from_numpy(sample)
    return sample

def predict(model, im_path, scale=20, threshold=0.5):
    scales = {
        80: 0,
        40: 1,
        20: 2
    }
    model.eval()
    sample = load_image(im_path)
    image = sample.cpu().numpy().reshape((640, 640))
    with torch.no_grad():
        outputs = model(sample.float())[scales[scale]][0]
        out0 = torch.sigmoid(outputs[0].cpu().detach()).numpy()
        out0[out0 < threshold] = 0
        out0[out0 >= threshold] = 1
        for i in range(scale):
            for j in range(scale):
                if out0[i][j] == 1:
                    x1 = j * 640 // scale
                    y1 = i * 640 // scale
                    x2 = (j + 1) * 640 // scale
                    y2 = (i + 1) * 640 // scale
                    x1, x2, y1, y2 = min(x1, 639), min(x2, 639), min(y1, 639), min(y2, 639)
                    image[y1:y2, x1] = 1
                    image[y1:y2, x2] = 1
                    image[y1, x1:x2] = 1
                    image[y2, x1:x2] = 1

        out1 = torch.sigmoid(outputs[1].cpu().detach()).numpy()
        out1[out1 < threshold] = 0
        out1[out1 >= threshold] = 1
        for i in range(scale):
            for j in range(scale):
                if out1[i][j] == 1:
                    x1 = j * 640 // scale
                    y1 = i * 640 // scale
                    x2 = (j + 1) * 640 // scale
                    y2 = (i + 1) * 640 // scale
                    x1, x2, y1, y2 = min(x1, 639), min(x2, 639), min(y1, 639), min(y2, 639)
                    image[y1:y2:3, x1] = 1
                    image[y1:y2:3, x2] = 1
                    image[y1, x1:x2:3] = 1
                    image[y2, x1:x2:3] = 1
    return image