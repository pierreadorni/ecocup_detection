import os
import csv
from matplotlib import pyplot as plt
from skimage import io
from alive_progress import alive_bar, alive_it



def main():
    csv_file = 'predictions_prepped.csv'
    images_bboxs = {}
    with open(csv_file, 'r') as f:
        r = csv.reader(f)
        for row in alive_it(r):
            if row[0] not in images_bboxs:
                images_bboxs[row[0]] = []
            images_bboxs[row[0]].append((int(row[1]), int(row[2]), int(row[3]), int(row[4]), float(row[5])))

    print("csv file loaded")
    for image_name in images_bboxs:
        image = io.imread(os.path.join('data/test', f"{int(image_name):03d}.jpg"))
        plt.imshow(image)
        for bbox in images_bboxs[image_name]:
            if bbox[4] < 0.6: continue
            plt.gca().add_patch(plt.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], fill=False, edgecolor='r', linewidth=2))
            plt.gca().text(bbox[1], bbox[0] - 2, f"{bbox[4]:.2f}", color='r')
        plt.show()

if __name__ == '__main__':
    main()