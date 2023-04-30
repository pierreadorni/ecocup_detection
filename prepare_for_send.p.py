import csv

with open("predictions_prepped.csv", 'w') as file:
    writer = csv.writer(file)
    with open("predictions.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            filename, y, x, h, w, confidence = row
            filename = int(filename.split('.')[0])
            if float(confidence) < 0.7: continue
            writer.writerow([filename, y, x, h, w, confidence])

