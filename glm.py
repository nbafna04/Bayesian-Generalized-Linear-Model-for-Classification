from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
import numpy as np
from numpy.linalg import inv
import warnings
import math
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Read all the Data files and labels
#The files are expected to be in the same directory where the python program would be placed

A = pd.read_csv(r"pp3data/A.csv", header=None)
A_label = pd.read_csv(r"pp3data/labels-A.csv", header=None)

AO = pd.read_csv(r"pp3data/AO.csv", header=None)
AO_label = pd.read_csv(r"pp3data/labels-AO.csv", header=None)

AP = pd.read_csv(r"pp3data/AP.csv", header=None)
AP_label = pd.read_csv(r"pp3data/labels-AP.csv", header=None)

USPS = pd.read_csv(r"pp3data/usps.csv", header=None)
USPS_label = pd.read_csv(r"pp3data/labels-usps.csv", header=None)

irls = pd.read_csv(r"irlstest.csv", header=None)
irls_label = pd.read_csv(r"labels-irlstest.csv", header=None)

from scipy.special import expit

#Calculate y, r and R for logistic model
def logistic_model(X, w):
    y = expit(np.dot(X, w))
    r = y * (1 - y)
    R = np.diagflat(r)
    return (y, R)

#Calculate error for logistic model using Wmap that was converged
def errorLogistic(wmap, phi):
    return expit(np.dot(phi, wmap))

#Calculate y, r and R for poisson model
def poisson_model(X,w):
    y = np.exp(np.dot(X,w))
    r = y
    R = np.diagflat(r)
    return (y, R)

#Function to calculate first derivate
def calculate_d(X, w, t):
    a = np.dot(X, w)
    d = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        d[i] = calculate_y(a[i], int(t[i])) + calculate_y(a[i], int(t[i]-1)) - 1
    d = d.reshape(t.shape[0],1)
    return d

#Calculate second derivative
def calculate_r(X, w, t):
    a = np.dot(X, w)
    r = np.zeros(t.shape[0])
    for i in range(t.shape[0]):
        r[i] = calculate_y(a[i], int(t[i]))*(1-calculate_y(a[i], int(t[i]))) + calculate_y(a[i], int(t[i]) - 1)*(1-calculate_y(a[i], int(t[i]) - 1))
    r = r.reshape(t.shape[0], 1)
    return np.diagflat(r)

#Recursive function to calculate yij
def calculate_y(a,j):
    phik = [-float('inf'), -2, -1, 0, 1, float('inf')]
    phi = np.array(phik)
    y = np.zeros(len(a))
    for i in range(len(a)):
        y[i] = expit(phi[j]- a[i])
    y.reshape(len(a), 1)
    return y

#Function to calculate data of Ordinal model
def ordinal_model(X,w,t):
    first_derivative = calculate_d(X,w,t)
    R = calculate_r(X, w, t)
    return first_derivative, R

#Function of Generalized linear model using Newton method
def glm_bayesian_logistic(X,test_data, t, test_label, model,alpha):

    #Start w with values 0
    w = np.zeros((np.shape(X)[1], 1))
    if model == 'logistic':
        y,R = logistic_model(X,w)
        first_derivative = y - t
        mat = np.dot(np.dot((X.T), R), (X))
        start_time = time.time()
        w_new = w - (
            np.dot(inv(alpha * np.identity(np.shape(mat)[0]) + mat), (np.dot(X.T, first_derivative) + alpha * w)))
    elif model == 'poisson':
        y,R = poisson_model(X,w)
        first_derivative = y - t
        mat = np.dot(np.dot((X.T), R), (X))
        start_time = time.time()
        w_new = w - (np.dot(inv(alpha * np.identity(np.shape(mat)[0]) + mat), (np.dot(X.T, first_derivative) + alpha * w)))
    elif model == 'ordinal':
        first_derivative, R = ordinal_model(X, w, t)
        mat = np.dot(np.dot((X.T), R), (X))
        start_time = time.time()
        w_new = w - (
            np.dot(inv(-1 * alpha * np.identity(np.shape(mat)[0]) - mat), (np.dot(X.T, first_derivative) - alpha * w)))
    else:
        print("Please select correct model")
        exit()

    n = 1
    #Till the value of w converges or till 100 iterations
    while (((LA.norm(w_new - w) / LA.norm(w)) >= 10 ** (-3)) and (n <= 100)):
        w = w_new
        if model == 'logistic':
            y, R = logistic_model(X, w)
            first_derivative = y - t
            mat = np.dot(np.dot((X.T), R), (X))
            w_new = w - (np.dot(inv(alpha * np.identity(np.shape(mat)[0]) + mat),
                                (np.dot(X.T, first_derivative) + alpha * w)))
        elif model == 'poisson':
            y, R = poisson_model(X, w)
            first_derivative = y - t
            mat = np.dot(np.dot((X.T), R), (X))
            w_new = w - (np.dot(inv(alpha * np.identity(np.shape(mat)[0]) + mat),
                                (np.dot(X.T, first_derivative) + alpha * w)))
        elif model == 'ordinal':
            #print(w)
            first_derivative, R = ordinal_model(X, w, t)
            mat = np.dot(np.dot((X.T), R), (X))
            w_new = w - (np.dot(inv(-1 * alpha * np.identity(np.shape(mat)[0]) - mat),
                                (np.dot(X.T, first_derivative) - alpha * w)))
        else:
            print("Please select correct model")
            exit()

        n += 1
    #print(w_new)
    end_time = time.time()
    timeForconvergence = end_time-start_time

    if model == 'logistic':
        A1 = errorLogistic(w_new, test_data)
        prediction = np.where(A1 >= 0.5, 1, 0)
        output = np.count_nonzero(prediction!=test_label)
    elif model == 'poisson':
        A1 = np.dot(test_data, w_new)
        lam = np.exp(A1)
        prediction = np.floor(lam)
        predicted = abs(prediction-test_label)
        output = len(test_label) - np.count_nonzero(predicted == 0)
    elif model == 'ordinal':

        A1 = np.dot(test_data, w_new)
        y = []
        phi = [-float('inf'), -2, -1, 0, 1, float('inf')]
        for j in range(0,6):
            y.append(expit((phi[j] - A1)))
        y = np.concatenate(y, axis=1)

        p = []
        for each in y:
            tmp = []
            for i in range(1,6):
                tmp.append(each[i] - each[i-1])
            p.append(tmp)
        p = np.array(p)
        prediction = np.argmax(p,axis =1)+1
        predicted = abs(prediction - test_label.flatten())
        output = len(test_label) - np.count_nonzero(predicted == 0)
    return output, n, timeForconvergence

#Function to randomly shuffle train test split
#Our test data is one third of the total data
def train_test_split(data_file, label_file, point):
    n = len(data_file)

    # N*1 matrix of values 1
    ones = pd.DataFrame(np.ones((np.shape(data_file)[0])))
    data_file = (pd.concat([ones,data_file],axis=1))
    index = np.arange(n)
    random.shuffle(index)

    data_file = data_file.values
    label_file = label_file.values

    # Split file in train,test data with test data as one third of total data
    test_index = index[:int(n / 3)]

    test_data = data_file[test_index]
    test_label = label_file[test_index]
    train_index = index[int(n / 3):]
    train_data = data_file[(train_index[:int(len(train_index) * point)])]
    train_label = label_file[(train_index[:int(len(train_index) * point)])]

    return train_data, test_data, train_label, test_label


#Function readying for the plotting of graphs
def plotFunction(filename , data_file, label_file, model):

    #Training Size from 0.1 to 1
    dataPoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alpha = 0.1
    training_Size = []
    averageErrorDict = {}
    stdErrorDict = {}
    avgtimeConvergence = []
    avgIterations = []
    timeDict = {}
    iterationDict = {}

    for point in dataPoints:
        averageError = []

        for i in range(30):
            #print("iteration", i)
            train_data, test_data,  train_label, test_label = train_test_split(data_file,label_file,point)
            output , iterations, timeForConvergence = glm_bayesian_logistic(train_data, test_data,
                                                                            train_label, test_label, model, alpha)
            averageError.append(output/len(test_label))
            avgtimeConvergence.append(timeForConvergence)
            avgIterations.append(iterations)
        training_Size.append(point*len(train_data))

        averageErrorDict[point] = np.mean(averageError)
        stdErrorDict[point] = np.std(averageError)
        iterationDict[point] = np.mean(avgIterations)
        timeDict[point] = np.mean(avgtimeConvergence)

    print("---------------------------------------------------")
    print("Statistics for model : ",model, " for dataset : ",filename)
    print("Average of number of iterations : ",iterationDict)
    print("Average runtime until convergence : ",timeDict)
    print("Mean error rate ",averageErrorDict)

    #print(training_Size)
    plt.gcf().clear()
    plt.errorbar(training_Size, averageErrorDict.values(), yerr=stdErrorDict.values(), ecolor='g', color='orange', capsize=20,
                         label=model)
    plt.xlabel("Train Size")
    plt.ylabel("Error Rate")
    plt.title("Dataset: " + filename)
    plt.savefig(filename + '.png')

#Function call for each dataset



plotFunction("A", A, A_label, 'logistic')
plotFunction("USPS",USPS, USPS_label, 'logistic')
plotFunction("AP", AP, AP_label, 'poisson')
plotFunction("irls", irls, irls_label, 'ordinal')
plotFunction("AO", AO, AO_label, 'ordinal')