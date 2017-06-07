import pandas as pd
import numpy as np
data = np.loadtxt("WholesaleCustomersData.csv", delimiter=',', skiprows=1)
inds = [i for i in range(2, data.shape[1])] + [1]
print(inds)
data = data[:, inds ]
print(data)

import csv
with open('WholesaleCustomersData.txt', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
