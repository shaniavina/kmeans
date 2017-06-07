from sklearn import preprocessing
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from numpy.linalg import lstsq
from numpy.linalg import norm
import csv
from scipy.stats import itemfreq
from matplotlib.pyplot import plot


def initiateBeta(num_of_variables,num_of_clusters, initialData):
    """
    initiate beta
    """
    #####simplex lattice design points
    weights = np.loadtxt("SLD5.txt", delimiter = " ")
    center = np.array([[1.0 / num_of_variables for i in range(num_of_variables)]])
    weights = np.concatenate((weights, center), axis=0)
    
    weights *= num_of_variables
    sqrtWeights = np.sqrt(weights)
    ys = []
    for weight in sqrtWeights:
        estimator = KMeans(num_of_clusters)
        data = initialData * weight
        estimator.fit_predict(data)
        minDistance = np.min(estimator.transform(data), 1)
        square = np.power(minDistance,2)
        sum_of_squares = np.sum(square)
        meanSquareDistance = sum_of_squares / (len(square) - 1)
        ys.append(meanSquareDistance)
    
    npys = np.array(ys)
    
    beta = lstsq(weights, npys)[0]
    
    return beta
    
def initiateAlpha(beta,num_of_variables):
    """
    initiate alpha
    """
    #####reduced variation

    RV = np.array((1 - beta) / np.sum(1 - beta))

    sum_of_RV = 0
        
    t = 0;    
    for r in RV:
        sum_of_RV += r
        if sum_of_RV > (1 - 1 / num_of_variables):
            break;
        t += 1
        
    t = min(t, num_of_variables - 1)
    if t == num_of_variables - 1:
        t -= 1

    g = []
    for i in range(num_of_variables):
        g.append((i + 1) * (beta[i] - np.mean(beta[:i + 1])) * (num_of_variables - 1) / (2 * num_of_variables))
    
    g = np.array(g)
        
    alpha = (g[t] + g[t + 1]) / 2
    
    return (alpha, t, g)

def initiateWeights(initialBeta,t,m,initialAlpha):
    initialWeights = []
    sum_of_beta = np.mean(initialBeta[:t + 1])
    for i in range(m):
        if i <= t:
            initialWeights.append(m / (t + 1) + (sum_of_beta -initialBeta[i]) * (m - 1) / (2 * initialAlpha) )
        else:
            initialWeights.append(0)
    
    initialWeights = np.array(initialWeights)
    
    return initialWeights
        
def processIter(weights,num_of_clusters,initialData ):
    sqrtWeights = np.sqrt(weights)

    data = initialData * sqrtWeights
    
    centroid, label, inertia = k_means(X = data, n_clusters = num_of_clusters, init='k-means++', precompute_distances=True, n_init=50)
    
    cents = np.zeros((num_of_clusters, initialData.shape[1]))
    
    counts = np.zeros((num_of_clusters, 1))
    
    for i in range(initialData.shape[0]):
        cents[label[i]] += initialData[i]
        counts[label[i]] += 1
        
    cents /= counts
    
    
    for i in range(initialData.shape[0]):
        data[i] = initialData[i] - cents[label[i]]
        
    data = np.power(data, 2)
        
    beta = np.mean(data, axis=0)
    
    return beta, label
    
    
def savePoints(weights, iter, data):
    data = data * np.sqrt(weights)

    with open("tmpData_" + str(iter) + ".txt", "w") as f:
        for i in range(5):
            for j in range(100):
                ind = i * 100 + j
                line = str(data[ind,0]) + "," + str(data[ind, 1]) + "," + str(data[ind, 2]) + "," + str(i) + "\n"
                f.write(line)
        
def rotateData(data, beta):
    betaSorted = sorted(enumerate(beta), key=lambda x:x[1])
    
    indices = []
    newBeta = []
    
    for b in betaSorted:
        indices.append(b[0])
        newBeta.append(b[1])
    
    data = data[:, indices]
    
    return data, np.array(newBeta), indices

def main():
    ####read data
    #data = np.loadtxt("/home/tongfei/Documents/LiClipse Workspace/shanshan/simulateData1.txt", delimiter = ",")
    f=open("whole.data.sorted.txt")
    reader=csv.reader(f)
    data = []
    label = []
    orderedLabel = []
    labelSet = set()
    for row in reader:
        if len(row) == 0:
            continue
        label.append(row[-1])
        if row[-1] not in labelSet:
            labelSet.add(row[-1])
            orderedLabel.append(row[-1])
        r = [float(n) for n in row[:-1]]    
        data.append(r)
    
    labelIndexMap = {}
    index = 0
    for l in orderedLabel:
        labelIndexMap[l] = index
        index += 1
            
    labelIndices = [labelIndexMap[l] for l in label]
    data = np.array(data)
    ####standardize
    data = preprocessing.scale(np.array(data))
    
    ####number of variables
    m = data.shape[1]
    
    ####number of clusters
    k = len(labelSet)
    
    #####initialize beta and alpha
    initialBeta = initiateBeta(m,k,data)
    
    data, initialBeta, indices = rotateData(data, initialBeta)
    
    
    
    result = initiateAlpha(initialBeta,m)
    initialAlpha = result[0]
    initial_t = result[1]
    initial_g = result[2]
    
    ####initial weights
    initialWeights = initiateWeights(initialBeta,initial_t, m,initialAlpha)
    currentWeights = initialWeights
    resLabel = labelIndices
    ####process with the iteration
    betaList = [initialBeta]
    tList = [initial_t]
    gList = [initial_g]
    alphaList = [initialAlpha]
    weightList = [initialWeights]
    dist = 1
    iter = 0
    maxIter = 10
    
    while dist  > 1e-6 and  iter < maxIter:
        #savePoints(initialWeights, iter, data)
        newBeta, resLabel = processIter(initialWeights, k, data )
        
        data, newBeta, inds = rotateData(data, newBeta)

        (newAlpha, newt, newg) = initiateAlpha(newBeta, m)
        
        betaList.append(newBeta)
        tList.append(newt)
        gList.append(newg)
        alphaList.append(newAlpha)
        
        dist = np.linalg.norm(newBeta - initialBeta)
        
        currentWeights = initialWeights
        initialWeights = initiateWeights(newBeta,newt, m,newAlpha)
        initialBeta = newBeta
        iter += 1
        weightList.append(currentWeights)
    
    print(betaList,alphaList,weightList,tList,iter)
    freq = itemfreq(labelIndices)
    freqMap = {}
    for f in freq:
        freqMap[int(f[0])] = int(f[1])
    
    resFreq = []
    begin = 0
    labelMap = {}
    labelComp = []
    begin
    while begin < len(labelIndices):
        label = int(labelIndices[begin])
        end = begin + freqMap[label]
        resFreq.append(itemfreq(resLabel[begin:end]))
        labelComp.append([labelIndices[begin:end],resLabel[begin:end]])
        maxCount = 0
        rightLabel = None
        for rf in resFreq[-1]:
            if rf[1] > maxCount:
                maxCount = rf[1]
                rightLabel = int(rf[0])
        labelMap[rightLabel] = int(label)
        begin = end

    resArray = []    
    for rf in resFreq:
        line = [0 for i in range(k)]
        for f in rf:
            line[int(labelMap[int(f[0])])] = int(f[1])
        resArray.append(line)
    for r in resArray:
        print(r)
        print("\n")
        
if __name__ == "__main__":
    '''
    alpha = 2 ** (-5)
    import math
    m = 8
    alphaWeights = [[] for i in range(m + 1)]
    while alpha < 2 ** 5:
        weights = np.array([])
        for t in range(m - 1, -1, -1):
            weights = initiateWeights(np.array([ 0.03113223,  0.03467171,  0.03710103,  0.98512094,  0.98719898, 0.99023446,  0.99359646,  0.99600486]), t, m, alpha)
            allNonNegtive = True
            for i in weights:
                if i < 0:
                    allNonNegtive = False
                    break
            if allNonNegtive:
                break
        alphaWeights[0].append(math.log(alpha) / math.log(2))
        for i in range(m):
            alphaWeights[i + 1].append(weights[i])
        alpha += 0.01
    import matplotlib.pyplot as plt
    for i in range(1, m + 1):
        plt.plot(alphaWeights[0], alphaWeights[i], label = "$w{i}$".format(i=i))
    plt.legend(loc='best')
    plt.show()'''
    
    main()
