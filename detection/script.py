import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import csv

def possible_objects(img):
    #copy image
    imOut = img.copy()
    # resize image
    newHeight = 200
    newWidth = int(imOut.shape[1]*200/imOut.shape[0])
    imOut = cv2.resize(imOut, (newWidth, newHeight))
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set input image on which we will run segmentation
    ss.setBaseImage(imOut)
    # Switch to slow but high recall Selective Search method
    # ss.switchToSelectiveSearchFast()
    ss.switchToSelectiveSearchQuality()
    # run selective search segmentation on input image
    rects = ss.process()
    # multiply the coordinates of the bounding boxes by the scale factor
    scale_factor_x = img.shape[1]/imOut.shape[1]
    scale_factor_y = img.shape[0]/imOut.shape[0]
    # create list of bboxes
    bboxes = []
    # iterate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal
        x, y, w, h = rect
        x, y, w, h = round(x*scale_factor_x), round(y*scale_factor_y), round(w*scale_factor_x), round(h*scale_factor_y)
        bboxes.append([x, y, w, h])
    return bboxes

def show_boxes(img, boxes):
    # copy image
    imOut = img.copy()
    # iterate over all the region proposals
    for rect in boxes:
        # draw rectangle for region proposal till 100
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (255, 0, 0), 1, cv2.LINE_AA)
    # show output
    plt.imshow(imOut)

def IoU(b1, b2):
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

# remove boxes with IoU > 0.5
def remove_overlapping_boxes(boxes, threshold = 0.5):
    # create list of bboxes
    bboxes = []
    # iterate over all the region proposals
    for i, rect in enumerate(boxes):
        # draw rectangle for region proposal
        x, y, w, h = rect
        # check if there is an overlap with a previous box
        overlap = False
        for j in range(i):
            if IoU(rect, boxes[j]) > threshold:
                overlap = True
                break
        if not overlap:
            bboxes.append([x, y, w, h])
    return bboxes

def select_region(filename, boxes):
        Imgs = []
        im = Image.open(filename)
        for i, rect in enumerate(boxes):
                x, y, w, h = rect
                im1 = im.crop((x, y, x+w, y+h))
                Imgs.append(im1)
        return Imgs

def show_label(filename):
    # Path to the directory containing the images
    image_dir = "images/pos/"

    # Load the image using PIL
    image_path = os.path.join(image_dir, filename + ".jpg")
    image = Image.open(image_path)

    # Path to the CSV file containing the crop values
    csv_dir = "labels_csv/"
    csv_path = os.path.join(csv_dir, filename + ".csv")

    # Load the CSV file using the csv module
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        crop_values_list = list(reader)

    for crop_values in crop_values_list:
        # Convert the crop values to integers
        crop_values = [int(value) for value in crop_values]
        print([crop_values[1],crop_values[0], crop_values[1]+crop_values[3], crop_values[0]+crop_values[2]])
        plt.imshow(image)
        plt.show()
        # Crop the image using the crop values
        cropped_image = image.crop((crop_values[1],crop_values[0], crop_values[1]+crop_values[3], crop_values[0]+crop_values[2]))
        # Show the cropped image
        cropped_image.show()
    return crop_values_list
#%%
filename = "images/pos/abigotte_pos_001.jpg"
boxes = possible_objects(cv2.imread(filename))
boxes = remove_overlapping_boxes(boxes, 0.8)
imgs = select_region(filename, boxes)
show_boxes(cv2.imread(filename), boxes)


#%%
import numpy as np
plt.figure(figsize=(round(np.sqrt(len(imgs)))+1,round(np.sqrt(len(imgs)))+1)) # specifying the overall grid size

for i in range(len(imgs)):
        plt.subplot(round(np.sqrt(len(imgs)))+1,round(np.sqrt(len(imgs)))+1,i+1)    # the number of images in the grid is 5*5 (25)
        plt.imshow(imgs[i])
        plt.title("img " + str(i))


#%%
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()
ax.imshow(cv2.imread(filename))
ax.add_patch(Rectangle((656, 325), 187, 182))
plt.show()


fig, ax = plt.subplots()
show_label("abigotte_pos_001")

#%%

for i in range(len(boxes)):
        if IoU([boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]], [709, 299, 844, 545]) > 0.75:
                print(i, IoU([boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]], [709, 299, 844, 545]))

print()
for i in range(len(boxes)):
        if IoU([boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]], [494, 297, 620, 550]) > 0.75:
                print(i, IoU([boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]], [494, 297, 620, 550]))

