import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random


class SOM():

    def __init__(self, filename, numClusters, learningRate) -> None:
        self.fileName = filename
        self.readFile()
        self.removeUniqueColumns()
        self.oneHotEncoding()
        self.scaleData()
        self.data = self.df.values.tolist()
        self.inputVectors = len(self.data[0])
        self.numClusters = numClusters
        self.weights = []
        self.learningRate = learningRate
        self.weightsDifference = []
    
    def readFile(self):
        df = pd.read_csv(self.fileName)
        self.df = df.dropna()
    
    def removeUniqueColumns(self):
        numberOfUnique = self.df.nunique()
        uniqueColumns = []
        for i in range(len(numberOfUnique)):
            columnName = self.df.columns[i]
            if numberOfUnique[i] == len(self.df) and not(self.df[columnName].dtype.kind in 'iufcb'):
                uniqueColumns.append(columnName)
        self.df.drop(uniqueColumns, axis=1, inplace=True)
        
    def oneHotEncoding(self):
        categoricalColumns = self.df.select_dtypes(include=["object"])
        categoricalColumnsLst = []
        for col in categoricalColumns:
            categoricalColumnsLst.append(col)
        self.df = pd.get_dummies(self.df, columns = categoricalColumnsLst)

    def scaleData(self):
        scaler = MinMaxScaler()
        df = self.df.copy()
        self.df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    def initializeWeights(self):
        for row in range(self.inputVectors):
            weights = []
            for column in range(self.numClusters):
                weight = random.random()
                weights.append(weight)
            self.weights.append(weights)
        # print(self.weights)

    def calculateDistances(self, inputVector):
        distances = []
        for column in range(self.numClusters):
            summation = 0
            for row in range(self.inputVectors):
                value = (inputVector[row] - self.weights[row][column])**2
                summation += value
            distances.append(summation)
        # print(distances)
        return distances

    def selectBestCluster(self, inputVector):
        distances = self.calculateDistances(inputVector)
        minDistance = min(distances)
        bestClusterIndex = distances.index(minDistance)
        # print(bestClusterIndex)
        return bestClusterIndex

    def updateWeights(self, inputVector, bestClusterIndex):
        for row in range(self.inputVectors):
            oldWeightValue = self.weights[row][bestClusterIndex]
            newWeightValue = oldWeightValue + self.learningRate*(inputVector[row]-oldWeightValue)
            difference = newWeightValue-oldWeightValue
            self.weightsDifference.append(difference)
            self.weights[row][bestClusterIndex] = newWeightValue
        



def SelfOrganizingMaps(filename, numClusters, learningRate, iterations):
    s = SOM(filename, numClusters, learningRate)
    s.initializeWeights()

    for iteration in range(iterations):
        print("********** Iteration Number = " + str(iteration+1) + " **********")
        for x in s.data:
            bestMatchingIndex = s.selectBestCluster(x)
            s.updateWeights(x, bestMatchingIndex)
            
        if (all(x <= 0.00000000001 for x in s.weightsDifference)):
            break
        else:
            s.learningRate = 0.5*s.learningRate
            print(s.weights)
            s.weightsDifference = []
    
    print("FINAL WEIGHTS:")
    print(s.weights)


filename = 'Self-OrganizingMaps/worldometer_coronavirus_summary_data.csv'
numClusters = 2
learningRate = 0.5
iterations = 100

SelfOrganizingMaps(filename, numClusters, learningRate, iterations)


# filename = 'Self-OrganizingMaps/worldometer_coronavirus_summary_data.csv'
# numClusters = 2
# inputVector = [0.0020963644071506204,0.007489443770728929,0.0019951958144534114,0.004823212496814616,0.13502464831068897,0.006058385453058164,0.034425319267073845,0.000930515331706835,0.0008408518256701044,0.028176465530080016,0.0,1.0,0.0,0.0,0.0,0.0]
# learningRate = 0.5
# s = SOM(filename, numClusters, learningRate)
# s.initializeWeights()
# b = s.selectBestCluster(inputVector)
# s.updateWeights(inputVector,b)
# print(s.weights)











