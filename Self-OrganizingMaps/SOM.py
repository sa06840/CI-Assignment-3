import random
from math import ceil, exp, log, sqrt
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class SOM():

    def __init__(self, filename, learningRate, iterations) -> None:
        self.fileName = filename
        self.readFile()
        self.replaceNan()
        self.removeUniqueColumns()
        self.oneHotEncoding()
        self.scaleData()
        self.data = self.df.values.tolist()
        self.inputVectors = len(self.data[0])
        self.getGridSize()
        self.weights = []
        self.learningRate = learningRate
        self.weightsDifference = []
        self.radius = self.gridSize//3
        self.timeConstant = log(iterations/self.radius)
        self.IDstoColor = []

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
        for columnName in uniqueColumns:
            name = str.lower(columnName)
            if "country" in name or "countries" in name:
                iDcolumn = columnName
                break
        self.uniquesIDs = self.df[iDcolumn].values.tolist() 
        self.df.drop(uniqueColumns, axis=1, inplace=True)

    def replaceNan(self):
        columnsWithNan = self.df.columns[self.df.isna().any()].tolist()
        for columnName in columnsWithNan:
            avg = self.df[columnName].mean()
            self.df[columnName] = self.df[columnName].fillna(avg)
          
    def oneHotEncoding(self):
        categoricalColumns = self.df.select_dtypes(include=["object"])
        categoricalColumnsLst = []
        for col in categoricalColumns:
            if "continent" in str.lower(col):
                self.df.drop([col], axis = 1, inplace=True)
            else:
                categoricalColumnsLst.append(col)
        if len(categoricalColumnsLst) != 0:
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

    def selectBestCluster(self, inputVector):
        minimum = [0,0]
        minimumDistance = 10000000000000
        for row in range(self.gridSize):
            for column in range(self.gridSize):
                weightList = self.weights[row][column]
                distance = 0
                for index in range(self.inputVectors):
                    value = (inputVector[index] - weightList[index])**2
                    distance += value
                if (distance < minimumDistance):
                    minimum[0] = row
                    minimum[1] = column
                    minimumDistance = distance
        return minimum
    
    def updateRadius(self, iterationNumber):
        newRadius = self.radius * exp(-(iterationNumber/self.timeConstant))
        self.radius = newRadius

    def neighbourhoodFunction(self, distance):
        num = distance**2
        den = 2*(self.radius**2)
        value = exp(-(num/den))
        return value

    def getEuclideanDistance(self, bestClusterIndex, weightListCoordinates):
        value = (weightListCoordinates[0]-bestClusterIndex[0])**2 + (weightListCoordinates[1]-bestClusterIndex[1])**2
        distance = sqrt(value)
        return distance

    def updateWeights(self, inputVector, bestClusterIndex):
        rowBestCluster = bestClusterIndex[0]
        columnBestCluster = bestClusterIndex[1]

        for row in range(self.gridSize):
            for column in range(self.gridSize):
                weightList = self.weights[row][column]
                if(row == rowBestCluster and column == columnBestCluster):
                    for weightIndex in range(self.inputVectors):
                        oldWeightValue = weightList[weightIndex]
                        newWeightValue = oldWeightValue + self.learningRate*(inputVector[weightIndex]-oldWeightValue)
                        difference = abs(newWeightValue-oldWeightValue)
                        self.weightsDifference.append(difference)
                        weightList[weightIndex] = newWeightValue
                    self.weights[row][column] = weightList

                else:
                    weightListCoordinates = [row,column]
                    distance = self.getEuclideanDistance(bestClusterIndex, weightListCoordinates)
                    if distance <= self.radius:
                        neighbourhoodValue = self.neighbourhoodFunction(distance)
                        for weightIndex in range(self.inputVectors):
                            oldWeightValue = weightList[weightIndex]
                            newWeightValue = oldWeightValue + self.learningRate*neighbourhoodValue*(inputVector[weightIndex]-oldWeightValue)
                            difference = abs(newWeightValue-oldWeightValue)
                            self.weightsDifference.append(difference)
                            weightList[weightIndex] = newWeightValue
                        self.weights[row][column] = weightList

    def assignCoordinatesID(self):
        xIndex = 0
        for x in self.data:
            bestClusterCoordinates = self.selectBestCluster(x)
            iD = self.uniquesIDs[xIndex]
            self.IDstoColor.append([iD, bestClusterCoordinates])
            xIndex+=1

    def getColorGridfromWeights(self):
        figure = plt.figure(figsize=(self.gridSize, self.gridSize))
        ax = figure.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.gridSize))
        ax.set_ylim((0, self.gridSize))
        for row in range(self.gridSize):
            for column in range(self.gridSize):
                rgb = [0, 0, 0] 
                weights = self.weights[row][column]
                for i in range(len(weights)):
                    if i % 3 == 0: 
                        rgb[0] = rgb[0] + weights[i]
                    elif i % 3 == 1: 
                        rgb[1] = rgb[1] + weights[i]
                    elif i % 3 == 2: 
                        rgb[2] = rgb[2] + weights[i]
              
                sumRGB = rgb[0] + rgb[1] + rgb[2]
                for i in range(len(rgb)):
                    rgb[i] = rgb[i]/sumRGB
                    
                ax.add_patch(plt.Rectangle((row, column), 1, 1, linewidth=0.5, antialiased=True, facecolor=(rgb[0], rgb[1], rgb[2], 1), edgecolor='black'))
                coordinates = [row, column]
                colour = [rgb[0], rgb[1], rgb[2], 1]
                label = ""
                for lst in self.IDstoColor:
                    if lst[1] == coordinates:
                        label = label + lst[0] + "\n"
                        lst.append(colour)
                ax.text(row+0.5, column+0.5, label, ha='center', va='center', color="white", fontsize=5)
        plt.show()
                    
    def getWorldMap(self):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        fig, ax = plt.subplots(figsize=(10, 10))
        world.plot(ax=ax, facecolor='lightgray', edgecolor='black')
        for lst in self.IDstoColor:
            iD = lst[0]
            color = lst[2]
            if iD in world["name"].tolist():
                world[world.name == iD].plot(color=color, ax=ax)
        plt.show()

        


def SelfOrganizingMaps(filename, learningRate, iterations):
    s = SOM(filename, learningRate, iterations)
    s.initializeWeights()

    for iteration in range(1,iterations+1):
        print("********** Iteration Number = " + str(iteration) + " **********")
        for x in s.data:
            bestMatchingIndex = s.selectBestCluster(x)
            s.updateWeights(x, bestMatchingIndex)
        print(bestMatchingIndex)
            
        if (all(w <= 0.00000000001 for w in s.weightsDifference)):
            break
        else:
            s.learningRate = 0.5*s.learningRate
            s.weightsDifference = []
            if (iteration%10 == 0):
                s.updateRadius(iteration)
    
    s.assignCoordinatesID()
    s.getColorGridfromWeights()
    s.getWorldMap()


# filename  = 'Self-OrganizingMaps/datasets/worldCoronaVirusData.csv'
# filename  = 'Self-OrganizingMaps/datasets/worldEnvironmentalData.csv'
# filename  = 'Self-OrganizingMaps/datasets/worldHappinessData.csv'
# filename  = 'Self-OrganizingMaps/datasets/worldPopulationData.csv'
filename  = 'Self-OrganizingMaps/datasets/worldVaccinationData.csv'
learningRate = 0.5
iterations = 100

SelfOrganizingMaps(filename, learningRate, iterations)
