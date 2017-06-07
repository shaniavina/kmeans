import csv
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = []
label = []
labelSet = set()

with open("simulateData2(1).txt") as f:
    reader=csv.reader(f)
    for row in reader:
        if len(row) == 0:
            continue
        label.append(row[-1])
        labelSet.add(row[-1])
        r = [float(n) for n in row[:-1]]    
        data.append(r)
labelIndexMap = {}
index = 0
for l in labelSet:
    labelIndexMap[l] = index
    index += 1
            
            
            
labelIndices = [labelIndexMap[l] for l in label]
    
data = preprocessing.scale(np.array(data))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colorMap = {0:'r', 1:'g', 2:'m', 3:'c', 4:'k', 5:'b', 6:'y'}
markerMap = {0:'o', 1:'v', 2:'^', 3:'<', 4:'>'}

b = 0
e = 100
for i in range(7):
    b = i * 200
    e = (i + 1) * 200
    ax.scatter(data[b:e,0], data[b:e,1], data[b:e,2], c = colorMap[i], marker = 'o')
    #b = e
    #e += (4 - i) * 20
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()









