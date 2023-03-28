from math import ceil, sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random


class SOM():

    def __init__(self, filename, learningRate) -> None:
        self.fileName = filename
        self.readFile()
        self.replaceNan()
        self.removeUniqueColumns()
        self.oneHotEncoding()
        self.scaleData()
        self.data = self.df.values.tolist()
        # print(self.data)
        self.inputVectors = len(self.data[0])
        self.getGridSize()
        self.weights = []
        self.learningRate = learningRate
        self.weightsDifference = []
    
    def readFile(self):
        df = pd.read_csv(self.fileName)
        self.df = df
    
    def removeUniqueColumns(self):
        numberOfUnique = self.df.nunique()
        uniqueColumns = []
        for i in range(len(numberOfUnique)):
            columnName = self.df.columns[i]
            if numberOfUnique[i] == len(self.df) and not(self.df[columnName].dtype.kind in 'iufcb'):
                uniqueColumns.append(columnName)
        self.df.drop(uniqueColumns, axis=1, inplace=True)

    def replaceNan(self):
        columnsWithNan = self.df.columns[self.df.isna().any()].tolist()
        for columnName in columnsWithNan:
            avg = self.df[columnName].mean()
            self.df[columnName] = self.df[columnName].fillna(avg)

        # self.df.to_csv("newfile.csv", index=False)
          
    def oneHotEncoding(self):
        categoricalColumns = self.df.select_dtypes(include=["object"])
        categoricalColumnsLst = []
        for col in categoricalColumns:
            categoricalColumnsLst.append(col)
        self.df = pd.get_dummies(self.df, columns = categoricalColumnsLst)

    def getGridSize(self):
        self.numClusters = 5*(sqrt((len(self.df))))
        self.gridSize = ceil(sqrt(self.numClusters))

    def scaleData(self):
        scaler = MinMaxScaler()
        df = self.df.copy()
        self.df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    def initializeWeights(self):
        for y in range(self.gridSize):
            row = []
            for x in range(self.gridSize):
                weights = []
                for i in range(self.inputVectors):
                    weight = random.random()
                    weights.append(weight)
                row.append(weights)
            self.weights.append(row)

        # print(self.weights)
        # print(len(self.weights))
        # for row in range(self.inputVectors):
        #     weights = []
        #     for column in range(self.numClusters):
        #         weight = random.random()
        #         weights.append(weight)
            
        #     self.weights.append(weights)
        # # print(self.weights)

    def selectBestCluster(self, inputVector):
        # distances = []
        minimum = [0,0]
        minimumDistance = 10000000000000
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                weightList = self.weights[y][x]
                distance = 0
                for index in range(self.inputVectors):
                    value = (inputVector[index] - weightList[index])**2
                    distance += value
                if (distance < minimumDistance):
                    minimum[0] = y
                    minimum[1] = x
                    minimumDistance = distance

        return minimum
    
                # distances.append(distance)

        # distances = []
        # for column in range(self.numClusters):
        #     summation = 0
        #     for row in range(self.inputVectors):
        #         value = (inputVector[row] - self.weights[row][column])**2
        #         summation += value
        #     distances.append(summation)
        # print(distances)
        # return distances

    # def selectBestCluster(self, inputVector):
    #     distances = self.calculateDistances(inputVector)
    #     minDistance = min(distances)
    #     bestClusterIndex = distances.index(minDistance)
    #     # print(bestClusterIndex)
    #     return bestClusterIndex


# IMPLEMENT NEIGHBOURHOOD FUNCTION

    def updateWeights(self, inputVector, bestClusterIndex):
        # y = bestClusterIndex//self.gridSize    #row
        # x = bestClusterIndex % self.gridSize   #column
        y = bestClusterIndex[0]
        x = bestClusterIndex[1]
        for weightIndex in range(self.inputVectors):
            oldWeightValue = self.weights[y][x][weightIndex]
            newWeightValue = oldWeightValue + self.learningRate*(inputVector[weightIndex]-oldWeightValue)
            difference = newWeightValue-oldWeightValue
            self.weightsDifference.append(difference)
            self.weights[y][x][weightIndex] = newWeightValue

        # for row in range(self.inputVectors):
        #     oldWeightValue = self.weights[row][bestClusterIndex]
        #     newWeightValue = oldWeightValue + self.learningRate*(inputVector[row]-oldWeightValue)
        #     difference = newWeightValue-oldWeightValue
        #     self.weightsDifference.append(difference)
        #     self.weights[row][bestClusterIndex] = newWeightValue
        



def SelfOrganizingMaps(filename, learningRate, iterations):
    s = SOM(filename, learningRate)
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
learningRate = 0.5
iterations = 100

SelfOrganizingMaps(filename, learningRate, iterations)


# filename = 'Self-OrganizingMaps/worldometer_coronavirus_summary_data.csv'
# numClusters = 2
# inputVector = [0.0020963644071506204,0.007489443770728929,0.0019951958144534114,0.004823212496814616,0.13502464831068897,0.006058385453058164,0.034425319267073845,0.000930515331706835,0.0008408518256701044,0.028176465530080016,0.0,1.0,0.0,0.0,0.0,0.0]
# learningRate = 0.5
# s = SOM(filename, learningRate)
# s.initializeWeights()
# s.calculateDistances(inputVector)
# b = s.selectBestCluster(inputVector)
# s.updateWeights(inputVector,b)
# print(s.weights)











