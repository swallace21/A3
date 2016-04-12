import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets

from sklearn import tree, svm, neighbors, cross_validation, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

from sklearn.cluster import MeanShift # as ms
from sklearn.datasets.samples_generator import make_blobs


activities = ['Driving', 'Running2', 'Stairs', 'Walking', 'WebBrowsing']
directory = '_DataFeaturesSmall/'

def featuresXY(filename):
    x = ''
    y = [filename] * 24
    xTest = ''
    yTest = [filename] * 6
    try:
        data = np.genfromtxt((directory+filename+'.csv'), delimiter=',')
        x = data[0:24,:]
        xTest = data[24:30,:]
    except Exception as e:
        print('Exception featuresXY: ', str(e))
    finally:
        return x, y, xTest, yTest


#Run through data...literally
Xdriving, Ydriving, XdrivingTest, YdrivingTest = featuresXY(activities[0])
Xrunning, Yrunning, XrunningTest, YrunningTest = featuresXY(activities[1]) #Running.csv I was wearing different pants
Xstairs, Ystairs, XstairsTest, YstairsTest = featuresXY(activities[2])
Xwalking, Ywalking, XwalkingTest, YwalkingTest = featuresXY(activities[3])
Xweb, Yweb, XwebTest, YwebTest = featuresXY(activities[4])


try:
    XtrainMast = np.array(Xdriving)
    XtrainMast = np.append(XtrainMast, Xrunning, 0)
    XtrainMast = np.append(XtrainMast, Xstairs, 0)
    XtrainMast = np.append(XtrainMast, Xwalking, 0)
    XtrainMast = np.append(XtrainMast, Xweb, 0)

    YtrainMast = np.array(Ydriving)
    YtrainMast = np.append(YtrainMast, Yrunning, 0)
    YtrainMast = np.append(YtrainMast, Ystairs, 0)
    YtrainMast = np.append(YtrainMast, Ywalking, 0)
    YtrainMast = np.append(YtrainMast, Yweb, 0)

    XtestMast = np.array(XdrivingTest)
    XtestMast = np.append(XtestMast, XrunningTest, 0)
    XtestMast = np.append(XtestMast, XstairsTest, 0)
    XtestMast = np.append(XtestMast, XwalkingTest, 0)
    XtestMast = np.append(XtestMast, XwebTest, 0)

    YtestMast = np.array(YdrivingTest)
    YtestMast = np.append(YtestMast, YrunningTest, 0)
    YtestMast = np.append(YtestMast, YstairsTest, 0)
    YtestMast = np.append(YtestMast, YwalkingTest, 0)
    YtestMast = np.append(YtestMast, YwebTest, 0)
except Exception as e:
    print('Exception append Arrays: ', str(e))


#Classifiers
clfLog = LogisticRegression().fit(XtrainMast,YtrainMast)
clfTree = tree.DecisionTreeClassifier().fit(XtrainMast,YtrainMast)
clfNbrs = KNeighborsClassifier(n_neighbors=5).fit(XtrainMast,YtrainMast)
clfSVM = svm.SVC(kernel='linear', C=1.0).fit(XtrainMast,YtrainMast)
#clfSVM = svm.SVC(kernel='poly', degree=1.5, C=1.0).fit(XtrainMast,YtrainMast)

#run basic scores and prediction
try:
    for act in activities:
        if act == activities[0]:
            xTest,yTest = XdrivingTest,YdrivingTest
        elif act == activities[1]:
            xTest,yTest = XrunningTest,YrunningTest
        elif act == activities[2]:
            xTest,yTest =  XstairsTest,YstairsTest
        elif act == activities[3]:
            xTest,yTest = XwalkingTest,YwalkingTest
        elif act == activities[4]:
            xTest,yTest = XwebTest,YwebTest

        print(act,' Basic Scores')
        print('log   = ', clfLog.score(xTest, yTest), clfLog.predict(xTest))
        print('tree  = ', clfTree.score(xTest, yTest), clfLog.predict(xTest))
        print('nbrs  = ', clfNbrs.score(xTest, yTest), clfLog.predict(xTest))
        print('svm   = ', clfSVM.score(xTest, yTest), clfLog.predict(xTest))
        #print('Score   log  = ', clfLog.score(xTest, yTest))
        #print('Predict log  = ', clfLog.predict(xTest))
        #print('Score   tree = ', clfTree.score(xTest, yTest))
        #print('Predict tree = ', clfTree.predict(xTest))
        #print('Score   nbrs = ', clfNbrs.score(xTest, yTest))
        #print('Predict nbrs = ', clfNbrs.predict(xTest))
        #print('Score   svm  = ', clfSVM.score(xTest, yTest))
        #print('Predict svm  = ', clfSVM.predict(xTest))
        print('')
except Exception as e:
    print('Exception runBasicScores: ', str(e))


def plot3D(title, Xlabel, Ylabel, Zlabel, inc, i):
    #inc = 4 to go through XYZ for given sensor
    #inc = 13 go through avg resultant acceleration
    #i = starting point
    ii = i + inc
    iii = ii = inc
    Xdr = np.genfromtxt((directory+activities[0]+'.csv'), delimiter=',')
    Xru = np.genfromtxt((directory+activities[1]+'.csv'), delimiter=',')
    Xst = np.genfromtxt((directory+activities[2]+'.csv'), delimiter=',')
    Xwa = np.genfromtxt((directory+activities[3]+'.csv'), delimiter=',')
    Xwe = np.genfromtxt((directory+activities[4]+'.csv'), delimiter=',')
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #print(Xdr[:, 1])
        xa = Xdr[:, i]
        ya = Xdr[:, ii]
        za = Xdr[:, iii]

        xb = Xru[:, i]
        yb = Xru[:, ii]
        zb = Xru[:, iii]

        xc = Xst[:, i]
        yc = Xst[:, ii]
        zc = Xst[:, iii]

        xd = Xwa[:, i]
        yd = Xwa[:, ii]
        zd = Xwa[:, iii]

        xe = Xwe[:, i]
        ye = Xwe[:, ii]
        ze = Xwe[:, iii]

        dr = ax.scatter(xa, ya, za, c='red', marker='o', label=activities[0])
        ru = ax.scatter(xb, yb, zb, c='orange', marker='o', label=activities[1])
        st = ax.scatter(xc, yc, zc, c='green', marker='o', label=activities[2])
        wa = ax.scatter(xd, yd, zd, c='purple', marker='o', label=activities[3])
        we = ax.scatter(xe, ye, ze, c='blue', marker='^', label=activities[4])

        ax.set_title(title)
        ax.set_xlabel('X '+ Xlabel)
        ax.set_ylabel('Y '+ Ylabel)
        ax.set_zlabel('Z '+ Zlabel)

        plt.legend(handles=[dr, ru, st, wa, we], loc=3)
        plt.show()
    except Exception as e:
        print('Exception plot3D: ', str(e))
    finally:
        return


def confusionMatrix(clfName, clf, Xtrain, Xtest, Ytrain, Ytest, color):
    try:
        clfRun = clf.fit(Xtrain, Ytrain).predict(Xtest)
        scores = cross_validation.cross_val_score(clf, Xtrain, Ytrain, cv=5)

        predicted = cross_validation.cross_val_predict(clf, Xtrain, Ytrain, cv=5)
        acc = metrics.accuracy_score(Ytrain, predicted)

        cm = confusion_matrix(Ytest, clfRun)
        cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure() #
        plt.imshow(cmNorm, interpolation='nearest', cmap=color)
        plt.title(clfName), plt.colorbar()
        tick_marks = np.arange(len(activities))
        plt.xticks(tick_marks, activities, rotation=45)
        plt.yticks(tick_marks, activities)
        plt.tight_layout()
        plt.ylabel('True'), plt.xlabel('Predicted')
        print(clfName +  ' Confusion Matrix')
        print('Scores: ',scores)
        print('Accuracy Metrics: %0.4f' % (acc))
        print('CM Norm: ',cmNorm)
        print ('Accuracy CrossValScore: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std() * 2))
        print('')
    except Exception as e:
        print('Exception confusionMatrix: ', str(e))


plot3D('Features 3D Plot', 'AccelX Stdev', 'AccelX Mean', 'Avg Resultant Accel (AccelX)', 1, 0) #inc, i,
#plot3D('Accelerometer', 'mean', 1, 0,'_DataRaw/')

#refresh clf for confusion matrix
clfLog = LogisticRegression()
clfTree = tree.DecisionTreeClassifier()
clfNbrs = KNeighborsClassifier(n_neighbors=1)
clfSVM = svm.SVC(kernel='linear', C=2.0)

confusionMatrix('Logistic Regression', clfLog, XtrainMast, XtestMast, YtrainMast, YtestMast, plt.cm.Blues)
confusionMatrix('Decision Tree', clfTree, XtrainMast, XtestMast, YtrainMast, YtestMast, plt.cm.Reds)
confusionMatrix('Nearest Neighbors', clfNbrs, XtrainMast, XtestMast, YtrainMast, YtestMast, plt.cm.Oranges)
confusionMatrix('Support Vector Machines', clfSVM, XtrainMast, XtestMast, YtrainMast, YtestMast, plt.cm.Greens)
plt.show() #print confusion matrixes
