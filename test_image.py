import os
import time

import warnings
from typing import Tuple, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from alive_progress import alive_bar, alive_it
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize

warnings.filterwarnings('ignore')
colors = {
    'red': '\033[31m',
    'green': '\033[32m',
    'reset': '\033[0m'
}


def IoU(b1, b2) -> float:
    i1, j1, h1, l1 = b1
    i2, j2, h2, l2 = b2
    x1 = max(i1, i2)
    y1 = max(j1, j2)
    x2 = min(i1 + h1, i2 + h2)
    y2 = min(j1 + l1, j2 + l2)
    if x1 > x2 or y1 > y2:
        return 0
    else:
        return (x2 - x1) * (y2 - y1) / (h1 * l1 + h2 * l2 - (x2 - x1) * (y2 - y1))


def predict_ecocups(image_path: str, min_confidence=0.9) -> List[Tuple[Tuple[int, int, int, int], float]]:
    # check image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError('Image not found: {}'.format(image_path))

    # load image
    img = io.imread(image_path)

    # generate crops using a sliding window
    window_sizes = [(600, 300), (300, 150), (200, 100), (100, 50)]  # y,x
    step = round(min(0.05*img.shape[0], 0.05*img.shape[1]))
    y_lens = [len(range(0, img.shape[0] - window_size[0], step)) for window_size in window_sizes]
    x_lens = [len(range(0, img.shape[1] - window_size[1], step)) for window_size in window_sizes]
    crops = []
    crops_bboxes = []
    with alive_bar(sum([xl * yl for yl, xl in zip(y_lens, x_lens)]), title="generating crops", title_length=16) as bar:
        for window_size in window_sizes:
            for y in range(0, img.shape[0] - window_size[0], step):
                for x in range(0, img.shape[1] - window_size[1], step):
                    crops.append(img[y:y + window_size[0], x:x + window_size[1]])
                    crops_bboxes.append((y, x, window_size[0], window_size[1]))
                    bar()

    # encode each crop to HoG
    hog_crops = []
    with alive_bar(len(crops), title="encoding crops", title_length=16) as bar:
        for crop in crops:
            # resize crop to 100x200
            crop = resize(crop, (100, 200), anti_aliasing=True)
            hog_crops.append(hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1))
            bar()

    # predict each crop
    predictions = []
    clf = joblib.load("svm_model_good_result.joblib")
    with alive_bar(0, title="predicting", title_length=16) as bar:
        predictions = [p[1] for p in clf.predict_proba(hog_crops)]
        bar()

    # filter out positive crops based on min confidence
    predictions = np.array(predictions)
    positive_crops_count = predictions.nonzero()[0].shape[0]
    predictions[predictions < min_confidence] = 0
    print(
        f"filtered out {positive_crops_count - predictions.nonzero()[0].shape[0]} positive crops based on min confidence level ({min_confidence}): {colors['red']} {positive_crops_count} {colors['reset']} -> {colors['green']} {predictions.nonzero()[0].shape[0]} {colors['reset']}")

    # filter out positive crops based on IoU
    positive_crops_count = predictions.nonzero()[0].shape[0]
    positive_crops = sorted(predictions.nonzero()[0], key=lambda x: predictions[x], reverse=True)
    for i in range(len(positive_crops)):
        for j in range(i + 1, len(positive_crops)):
            if IoU(crops_bboxes[positive_crops[i]], crops_bboxes[positive_crops[j]]) > 0.5:
                predictions[positive_crops[j]] = 0
    print(
        f"filtered out {positive_crops_count - predictions.nonzero()[0].shape[0]} positive crops based on IoU: {colors['red']} {positive_crops_count} {colors['reset']} -> {colors['green']} {predictions.nonzero()[0].shape[0]} {colors['reset']}")
    positive_crops = predictions.nonzero()[0]

    return [(crops_bboxes[pos_crop_i], predictions[pos_crop_i]) for pos_crop_i in positive_crops]


def predict_folder(folder: str):
    images_folder = 'data/test'
    with open('predictions2.csv', 'r') as f:
        content = f.read()

    for image_name in alive_it(os.listdir(images_folder)):
        if image_name in content:
            print(f"skipping {image_name}")
            continue
        try:
            predictions = predict_ecocups(os.path.join(images_folder, image_name), min_confidence=0)
            image_predictions = [(image_name, *prediction[0], prediction[1]) for prediction in predictions]
            with open('predictions2.csv', 'a') as f:
                for prediction in image_predictions:
                    f.write(','.join([str(p) for p in prediction]) + '\n')
        except Exception as e:
            print(f"error predicting {image_name}: {e}")


if __name__ == '__main__':
    predict_folder('data/test')