#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:02:07 2023

@author: thomas
"""

import os
import cv2
import csv
from matplotlib import pyplot as plt
from typing import Tuple, Generator
import numpy as np
from skimage import io, util, transform, feature
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from datetime import datetime
import uuid
import sys


# concatenate all csv files into one
with open('data/labels.csv', 'w') as outfile:
    for file in os.listdir('data/labels_csv'):
        with open('data/labels_csv/' + file) as infile:
            # add filename to each line
            for line in infile:
                outfile.write(file[:-4] + ',' + line)

def get_ecocup_info(difficult: bool) -> Generator[Tuple[str, int, int, int, int], None, None]:
    with open("data/labels.csv") as file:
        line = file.readline()
        while line:
            img_name, x, y, w, h, diff = map(str.strip, line.split(','))
            if not difficult and diff == '1':
                # skip difficult images
                pass
            elif int(w) / int(h) < 1.5 or int(w) / int(h) > 2:
                # skip images with wrong aspect ratio
                pass
            else:
                yield img_name, x, y, w, h
            line = file.readline()

def resize_image(source: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(source, size)

def ecocup_image(image_file: str, x: int, y: int, w: int, h: int) -> np.ndarray:
    return io.imread('data/images/pos/' + image_file + '.jpg')[int(x):int(x) + int(w), int(y):int(y) + int(h)]


def get_pos_images(difficult=False) -> Generator[np.ndarray, None, None]:
    # generate positive images (ecocups)
    for info in get_ecocup_info(difficult):
        yield info, resize_image(ecocup_image(*info), (50, 100))

def save_pos_images(save_dir: str = 'data/augmented_images', difficult: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (info, image) in enumerate(get_pos_images(difficult)):
        filename = f"{i:04}.jpg"
        filepath = os.path.join(save_dir, filename)
        io.imsave(filepath, util.img_as_ubyte(image))

def augment_pos_images(num_images: int = 10, save_dir: str = 'data/augmented_images', difficult: bool = False):
        save_pos_images(save_dir, difficult)
        random.seed(datetime.now())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        i = len(os.listdir(save_dir))
        for info, image in get_pos_images(difficult):
            for j in range(num_images):
                # random flip
                if random.uniform(0,1) < 0.5:
                    image = cv2.flip(image, 1)
                if random.uniform(0,1) < 0.9:
                    augmented = image[
                    random.randint(0, int(image.shape[0]/4)):
                    random.randint(image.shape[0] - int(image.shape[0]/4), image.shape[0]),
                    random.randint(0, int(image.shape[1]/4)):
                    random.randint(image.shape[1] - int(image.shape[1]/4), image.shape[1])] # Slicing to crop the image
                # random contrast factor between 0.75 and 1.25
                contrast = random.uniform(0.85, 1.15)
                # random brightness shift between -25 and 25
                brightness = random.uniform(-20, 20)
                # random gamma correction factor between 0.75 and 1.25
                gamma = random.uniform(0.85, 1.15)
                # apply color changes
                augmented = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
                augmented = np.power(augmented / 255.0, gamma)
                augmented = np.uint8(augmented * 255)
                # save augmented image
                filename = f"{i:04}.jpg"
                filepath = os.path.join(save_dir, filename)
                io.imsave(filepath, util.img_as_ubyte(augmented))
                i += 1

def generate_negative_images(num_images: int, save_dir: str = 'data/negative_images', source_dir: str = 'data/images/neg'):
    random.seed(datetime.now())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    i = len(os.listdir(save_dir))
    for j in range(num_images):
        # choose a random source image
        source_file = random.choice(os.listdir(source_dir))
        source_image = io.imread(os.path.join(source_dir, source_file))
        # choose a random patch from the source image
        rand = random.randint(50,150)
        patch_size = (int(random.uniform(1.5, 2)*rand), rand)
        patch_x = random.randint(0, max(0,source_image.shape[0] - patch_size[0]))
        patch_y = random.randint(0, max(0,source_image.shape[1] - patch_size[1]))
        patch = source_image[patch_x:patch_x + patch_size[0], patch_y:patch_y + patch_size[1]]
        patch = transform.resize(patch, (200, 100))
        # save negative image
        filename = f"{i:04}.jpg"
        filepath = os.path.join(save_dir, filename)
        io.imsave(filepath, util.img_as_ubyte(patch))
        i += 1

def train_svm_with_hog(pos_dir = 'data/augmented_images', neg_dir = 'data/negative_images', model_dir = 'data/svm_model.joblib'):
    # Load positive images
    print("loading pos img")
    pos_images = []
    for filename in os.listdir(pos_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(pos_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            pos_images.append(hog_features)

    # Load negative images
    print("loading neg img")
    neg_images = []
    for filename in os.listdir(neg_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(neg_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            neg_images.append(hog_features)

    # Combine positive and negative images into a single dataset
    images = np.concatenate([pos_images, neg_images])
    labels = np.concatenate([np.ones(len(pos_images)), np.zeros(len(neg_images))])

    # split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    print("train model")
    # train the SVM
    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)

    # evaluate the model on the validation set
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.2f}")

    # save the trained model
    joblib.dump(clf, model_dir)

    return clf

def train_svm_poly(pos_dir = 'data/augmented_images', neg_dir = 'data/negative_images', model_dir = 'data/poly_model.joblib'):
    # Load positive images
    pos_images = []
    for filename in os.listdir(pos_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(pos_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            pos_images.append(hog_features)

    # Load negative images
    neg_images = []
    for filename in os.listdir(neg_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(neg_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            neg_images.append(hog_features)

    # Combine positive and negative images into a single dataset
    images = np.concatenate([pos_images, neg_images])
    labels = np.concatenate([np.ones(len(pos_images)), np.zeros(len(neg_images))])

    # split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    print("train model")
    # train the SVM
    clf = svm.SVC(probability=True, kernel='poly', degree=5)
    clf.fit(X_train, y_train)

    # evaluate the model on the validation set
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.2f}")

    # save the trained model
    joblib.dump(clf, model_dir)

    return clf

def train_rf_with_hog(pos_dir = 'data/augmented_images', neg_dir = 'data/negative_images', model_dir = 'data/rf_model.joblib'):
    # Load positive images
    print("loading pos img")
    pos_images = []
    for filename in os.listdir(pos_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(pos_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            pos_images.append(hog_features)

    # Load negative images
    print("loading neg img")
    neg_images = []
    for filename in os.listdir(neg_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(neg_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            neg_images.append(hog_features)

    # Combine positive and negative images into a single dataset
    images = np.concatenate([pos_images, neg_images])
    labels = np.concatenate([np.ones(len(pos_images)), np.zeros(len(neg_images))])

    # split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    print("train model")
    # train the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # evaluate the model on the validation set
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.2f}")

    # save the trained model
    joblib.dump(clf, model_dir)

    return clf

def train_adaboost_with_hog(pos_dir = 'data/augmented_images', neg_dir = 'data/negative_images', model_dir = 'data/adaboost_model.joblib'):
    # Load positive images
    print("loading pos img")
    pos_images = []
    for filename in os.listdir(pos_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(pos_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            pos_images.append(hog_features)

    # Load negative images
    print("loading neg img")
    neg_images = []
    for filename in os.listdir(neg_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(neg_dir, filename)
            image = io.imread(image_path)
            # resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            neg_images.append(hog_features)

    # Combine positive and negative images into a single dataset
    images = np.concatenate([pos_images, neg_images])
    labels = np.concatenate([np.ones(len(pos_images)), np.zeros(len(neg_images))])

    # split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    print("train model")
    # train the AdaBoost classifier
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # evaluate the model on the validation set
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.2f}")

    # save the trained model
    joblib.dump(clf, model_dir)

    return clf

def train_gboost_with_hog(pos_dir = 'data/augmented_images', neg_dir = 'data/negative_images', model_dir = 'data/gboost_model.joblib'):
    # Load positive images
    print("Loading positive images...")
    pos_images = []
    for filename in os.listdir(pos_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(pos_dir, filename)
            image = io.imread(image_path)
            # Resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # Compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            pos_images.append(hog_features)

    # Load negative images
    print("Loading negative images...")
    neg_images = []
    for filename in os.listdir(neg_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(neg_dir, filename)
            image = io.imread(image_path)
            # Resize the image to (100, 200)
            image = transform.resize(image, (100, 200))
            # Compute HOG features for the image
            hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
            neg_images.append(hog_features)

    # Combine positive and negative images into a single dataset
    images = np.concatenate([pos_images, neg_images])
    labels = np.concatenate([np.ones(len(pos_images)), np.zeros(len(neg_images))])

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the Gradient Boosting model
    print("Training Gradient Boosting model...")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=5)
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.2f}")

    # Save the trained model
    joblib.dump(clf, model_dir)

    return clf

def IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # calculate intersection area
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # calculate IoU
    if union_area == 0:
        iou = 0
    else:
        iou = intersection_area / union_area

    return iou

def get_box_from_labels(image_name, labels_file='data/labels.csv'):
    """Returns the list of bounding boxes for the given image_name in the labels_file."""
    #print("file", image_name)
    with open(labels_file, 'r') as f:
        reader = csv.reader(f)
        #next(reader)  # skip header row
        boxes = []
        for row in reader:
            if row[0] == image_name:
                y, x, h, w, s = map(int, row[1:])
                boxes.append((x, y, w, h))
        return boxes

def test_model_sw(classifier, test_images_dir='data/test', save_dir='data/results', result_file='data/result.csv', debug=False):
        clf = joblib.load(classifier)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(result_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            filenames = os.listdir(test_images_dir)
            filenames.sort()

            for n, filename in enumerate(filenames):
                if not filename.endswith('.jpg'):
                    continue
                print(filename)
                image_path = os.path.join(test_images_dir, filename)
                image = cv2.imread(image_path)
                img_height, img_width = image.shape[:2]
                true_height, true_width = image.shape[:2]

                window_factor = 15
                window_size = (int(min(img_width,img_height)/window_factor),int(2 * min(img_width,img_height)/window_factor))
                window_size = (100, 200)
                boxes_found = []
                sizes = [70, 100]
                ratios = [1.5, 1.75, 2]

                for size in sizes:
                    for ratio in ratios:
                        window_size = (int(size), int(size * ratio))
                        image = cv2.imread(image_path)
                        for k in range(8):
                            img_height, img_width = image.shape[:2]
                            w, h = window_size
                            for i in range(0, img_height - window_size[1], int(window_size[1]/3)):
                                for j in range(0, img_width - window_size[0], int(window_size[0]/3)):
                                    x, y = j, i
                                    proposal_img = image[y:y+h,x:x+w]
                                    if proposal_img.shape != (h, w, 3):
                                        break
                                    # resize the image to (100, 200)
                                    proposal_img = cv2.resize(proposal_img, (100, 200))
                                    # encode the image as HOG features
                                    hog_features = feature.hog(proposal_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
                                    # make a prediction using the SVM
                                    proba = clf.predict_proba(hog_features.reshape(1, -1))
                                    if proba[0][1] > 0.55:
                                        scale_x = true_width / img_width
                                        scale_y = true_height / img_height
                                        true_x = int(x * scale_x)
                                        true_y = int(y * scale_y)
                                        true_w = int(w * scale_x)
                                        true_h = int(h * scale_y)
                                        box = (true_x, true_y, true_w, true_h)
                                        boxes_found.append([box, proba[0][1]])

                            image = cv2.resize(image, (int(img_width*0.85), int(img_height*0.8)))
                image = cv2.imread(image_path)
                # sort boxes by probability
                boxes_found = sorted(boxes_found, key=lambda x: x[1], reverse=True)

                # loop through boxes and remove those with high IoU
                boxes_to_remove = []
                for i in range(len(boxes_found)):
                    for j in range(i+1, len(boxes_found)):
                        iou = IoU(boxes_found[i][0], boxes_found[j][0])
                        if iou > 0.4:
                            boxes_to_remove.append(j)
                boxes_to_remove = list(set(boxes_to_remove))
                boxes_found = [box for i, box in enumerate(boxes_found) if i not in boxes_to_remove]

                # loop through remaining boxes and test against labels
                if debug:
                    name, ext = os.path.splitext(filename)
                    labels = get_box_from_labels(name)
                    for box in boxes_found:
                        has_label = False
                        for label in labels:
                            iou = IoU(box[0], label)
                            if iou >= 0.3:
                                has_label = True
                                break
                        if not has_label:
                            #if random.uniform(0,1) < 0.5:
                            # Add image to negative training dataset
                            negative_img = image[box[0][1]:box[0][1]+box[0][3], box[0][0]:box[0][0]+box[0][2]]
                            negative_img = cv2.resize(negative_img, (100, 200))
                            cv2.imwrite(os.path.join('data/negative_images', f"false_positive_{str(uuid.uuid4())[:8]}.jpg"), negative_img)
                        else:
                            pos_img = image[box[0][1]:box[0][1]+box[0][3], box[0][0]:box[0][0]+box[0][2]]
                            pos_img = cv2.resize(pos_img, (100, 200))
                            cv2.imwrite(os.path.join('data/augmented_images', f"pos_{str(uuid.uuid4())[:8]}.jpg"), pos_img)


                # write the non-overlapping boxes to the CSV file
                for box in boxes_found:
                    x, y, w, h = box[0]
                    score = box[1]
                    writer.writerow([n, y, x, h, w, score])
                # draw the non-overlapping boxes on the image
                for box in boxes_found:
                    x, y, w, h = box[0]
                    score = box[1]
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, int((score-0.55) * 255*45), 0), 2)
                # save the image with boxes drawn on it
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, image)


if len(sys.argv) < 4:
    print('Utilisation: python ecocup_detection.py train <svm/poly/rf/gboost/adaboost> <positive_data_dir> <negative_data_dir> <model_file')
    print('             python ecocup_detection.py test <classifier.joblib> <test_data_dir> <output_dir> <results.csv>')
    print('             python ecocup_detection.py generate <pos> <num_images> <save_dir>')
    print('             python ecocup_detection.py generate <neg> <num_images> <save_dir> <source_dir>')
    sys.exit()

if sys.argv[1] == 'train':
        if sys.argv[2] == 'svm':
            clf = train_svm_with_hog(sys.argv[3], sys.argv[4], sys.argv[5])
            print('Le modèle à été sauvegardé sous', sys.argv[5])
        elif sys.argv[2] == 'poly':
            clf = train_svm_poly(sys.argv[3], sys.argv[4], sys.argv[5])
            print('Le modèle à été sauvegardé sous', sys.argv[5])
        elif sys.argv[2] == 'rf':
            clf = train_rf_with_hog(sys.argv[3], sys.argv[4], sys.argv[5])
            print('Le modèle à été sauvegardé sous', sys.argv[5])
        elif sys.argv[2] == 'gboost':
            clf = train_gboost_with_hog(sys.argv[3], sys.argv[4, sys.argv[5]])
            print('Le modèle à été sauvegardé sous', sys.argv[5])
        elif sys.argv[2] == 'adaboost':
            clf = train_gboost_with_hog(sys.argv[3], sys.argv[4, sys.argv[5]])
            print('Le modèle à été sauvegardé sous', sys.argv[5])
        else:
            print('Commande invalide. Utilisez "svm", "poly", "rf", "gboost" ou "adaboost".')
            sys.exit()

elif sys.argv[1] == 'test':
        if len(sys.argv) != 6:
            print('Utilisation: python ecocup_detection.py test <classifier.joblib> <test_data_dir> <output_dir> <results.csv>')
            sys.exit()

        clf = sys.argv[2]
        test_dir = sys.argv[3]
        output_dir = sys.argv[4]
        output_file = sys.argv[5]
        test_model_sw(clf, test_dir, output_dir, output_file)
        print(f'Les prédictions ont été sauvegardé sous {output_file}')

elif sys.argv[1] == 'generate':
        if sys.argv[2] == 'pos':
                print('Utilisation: python ecocup_detection.py generate pos <num_images> <save_dir>')
                augment_pos_images(int(sys.argv[3]), sys.argv[4])
        if sys.argv[2] == 'neg':
                print('Utilisation: python ecocup_detection.py generate <neg> <num_images> <save_dir> <source_dir>')
                generate_negative_images(int(sys.argv[3]), sys.argv[4], sys.argv[5])

else:
        print('Commande invalide. Utilisez "train", "test" or "generate".')
        sys.exit()
