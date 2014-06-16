import cPickle as cp
import numpy as np
import csv

reader = csv.reader(open("trainLabels.csv"), delimiter=',')
out = []
for row in reader:
    out.append(row)
names = out[1:]

key_num = 0
my_dict = {}
for elem in names:
    val = elem[1]
    if val not in my_dict:
        my_dict[val] = key_num
        key_num += 1
    if key_num == 10: break

labels = []
for elem in names:
        labels.append(my_dict[elem[1]])

cp.dump(np.array(labels, dtype=np.uint8), open("Train_labels", 'wb'), 2)
