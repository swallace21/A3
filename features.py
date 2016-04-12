from sklearn import preprocessing
import csv
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
#import pandas as pd

#variables
featuresCount = 78
featuresCountSmall = 5

scaler = preprocessing.MinMaxScaler([-1,1])

def arrayTut():
    #p = np.array([[1,2],[3,4]])
    p = np.zeros(shape=(1, 18))
    print('1 = ', p)
    p = np.append(p, [[5,6]], 0)
    print('2 = ', p)
    p = np.append(p, [[7],[8],[9]], 1)
    print('3 = ', p)
    a = [10,11]
    a.extend([12])
    print('a = ', a)
    p = np.append(p, [a], 0)
    print('4 = ', p)
    print('FINAL = ', p[3, :])

#each csv for training data contains 2400 data points, 10 secs per feature = 24 per csv
def newDataCSV(filename,location):
    tempList = [0] * featuresCount
    X_temp = np.array([tempList])  #adds zeros first so we can manipulate the array
    try:
        data = np.genfromtxt(('_DataRaw/'+filename), delimiter=',', skip_header=1)
        #print(data[0:10, 0]) #prints rows 0 to 10 for column 0

        rowRange = 3000

        rate = 100
        rows = rate
        i = 0
        #for row in range(rowRange):
        while rows <= rowRange:
            col = 0
            fRowTemp = []
            while col <= 17: #18*4 = 72
                dataEX = data[(rows-rate):rows, col]
                fRowTemp.extend(newFeatures(dataEX))
                check = col + 1
                if col % 3 == 0 and col != 0:
                    avgsqrt = avgSqrt(data[(rows-rate):rows, (col-3):col])
                    fRowTemp.extend(avgsqrt)
                col += 1
            #16-18
            avgsqrt = avgSqrt(data[(rows-rate):rows, (col-3):col])
            fRowTemp.extend(avgsqrt)
            #final part
            rows =  rows + rate
            X_temp = np.append(X_temp, [fRowTemp], 0)
            fRowTemp = []

        X_temp = np.delete(X_temp, (0), axis=0) #gets rid of row of zeros
        np.savetxt(location+filename, X_temp, delimiter=",") #works
    except Exception as e:
        print("Exception newDataCSV: ", filename, ' ERROR: ', str(e))
    finally:
        return print('Success for ', filename)


def newFeatures(dataNF):
    try:
        #proc = preprocessing.data #get standard dev and med
        cMin = min(dataNF)
        cMax = max(dataNF)
        cStdev = np.std(dataNF)
        cMean = np.mean(dataNF)
    except Exception as e:
        print('dataNF = ', dataNF[0,0])
        print("Exception newFeatures: ", str(e))
    finally:
        return [cMin, cMax, cStdev, cMean]
        #return [cStdev, cMean]

#each csv for training data contains 2400 data points, 10 secs per feature = 24 per csv
def newDataCSVSmall(filename,location):
    tempList = [0] * featuresCountSmall
    X_temp = np.array([tempList])  #adds zeros first so we can manipulate the array
    try:
        data = np.genfromtxt(('_DataRaw/'+filename), delimiter=',', skip_header=1)
        #print(data[0:10, 0]) #prints rows 0 to 10 for column 0

        rowRange = 3000

        rate = 100
        rows = rate
        i = 0
        #for row in range(rowRange):
        while rows <= rowRange:
            col = 0
            rM = (rows-rate)
            fRowTemp = []

            #Accel X
            dataEX = data[rM:rows, 0]
            #fRowTemp.extend([min(dataEX)])
            #fRowTemp.extend([max(dataEX)])
            fRowTemp.extend([np.std(dataEX)])
            fRowTemp.extend([np.mean(dataEX)])

            #Average Resultant Acceleration
            avgsqrt = avgSqrt(data[rM:rows, 0:3])
            fRowTemp.extend(avgsqrt)

            #Gravity Y (mean)
            dataEX = data[rM:rows, 4]
            fRowTemp.extend([np.mean(dataEX)])

            #Lin Accel Z (min, max)
            dataEX = data[rM:rows, 8]
            #fRowTemp.extend([min(dataEX)])
            #fRowTemp.extend([max(dataEX)])

            #Average Resultant Linear Acceleration
            avgsqrt = avgSqrt(data[rM:rows, 5:8])
            #fRowTemp.extend(avgsqrt)

            #Gyroscope Y (stdev)
            dataEX = data[rM:rows, 10]
            fRowTemp.extend([np.std(dataEX)])
            #fRowTemp.extend([(max(dataEX)+100)-(min(dataEX)+100)])

            #Average Resultant Gyroscope Y
            avgsqrt = avgSqrt(data[rM:rows, 9:12])
            #fRowTemp.extend(avgsqrt)

            #Gyroscope Z (stdev, min, max)
            dataEX = data[rM:rows, 11]
            #fRowTemp.extend([min(dataEX)])
            #fRowTemp.extend([max(dataEX)])
            ##fRowTemp.extend([(max(dataEX)+100)-(min(dataEX)+100)])
            ##fRowTemp.extend([np.std(dataEX)])
            ##fRowTemp.extend([np.mean(dataEX)])

            #Orientation Z (mean, min, max)
            dataEX = data[rM:rows, 15]
            #fRowTemp.extend([min(dataEX)])
            #fRowTemp.extend([max(dataEX)])
            #fRowTemp.extend([np.mean(dataEX)])
            #fRowTemp.extend([np.std(dataEX)])
            ##fRowTemp.extend([(max(dataEX)+100)-(min(dataEX)+100)])

            #final part
            rows =  rows + rate
            X_temp = np.append(X_temp, [fRowTemp], 0)
            fRowTemp = []

        X_temp = np.delete(X_temp, (0), axis=0) #gets rid of row of zeros
        np.savetxt(location+filename, X_temp, delimiter=",") #works
    except Exception as e:
        print("Exception newDataCSVSmall: ", filename, ' ERROR: ', str(e))
    finally:
        return print('Success for ', filename)

def avgSqrt(dataAS):
    try:
        X = sum(dataAS[:, 0])
        X = X*X
        Y = sum(dataAS[:, 1])
        Y = Y*Y
        Z = sum(dataAS[:, 2])
        Z = Z*Z
        fin = (np.sqrt(X+Y+Z))/100.0 #100.0 num of samples
    except Exception as e:
        print("Exception avgSqrt: ", str(e))
    finally:
        fin = fin.tolist()
        return [fin]


###############################################################################
#compute and then add features for all training data
#newDataCSV("Driving.csv", '_DataFeatures/')
#newDataCSV("Running.csv", '_DataFeatures/')
#newDataCSV("Running2.csv", '_DataFeatures/')
#newDataCSV("Stairs.csv", '_DataFeatures/')
#newDataCSV("Walking.csv", '_DataFeatures/')
#newDataCSV("Walking2.csv", '_DataFeatures/')
#newDataCSV("WebBrowsing.csv", '_DataFeatures/')

newDataCSVSmall("Driving.csv", '_DataFeaturesSmall/')
newDataCSVSmall("Running.csv", '_DataFeaturesSmall/')
newDataCSVSmall("Running2.csv", '_DataFeaturesSmall/')
newDataCSVSmall("Stairs.csv", '_DataFeaturesSmall/')
newDataCSVSmall("Walking.csv", '_DataFeaturesSmall/')
newDataCSVSmall("Walking2.csv", '_DataFeaturesSmall/')
newDataCSVSmall("WebBrowsing.csv", '_DataFeaturesSmall/')
