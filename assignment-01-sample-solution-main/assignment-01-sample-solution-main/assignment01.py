# -*- coding: utf-8 -*-
"""
File:   assignment01.py
Author: Khaled HAmad
Date:   09/13/2021
Desc:   Assignment 01 solution
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import FormatStrFormatter
plt.close('all') #close any open plots


""" ======================  Function definitions ========================== """

def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
     
        
        
        
     
""" ======================  Variable Declaration ========================== """
M = 20 #you can change the M value here for different implementations
s =0.5 #you can change the s value here for different implementations
""" =======================  Load Training Data ======================= """
data_uniform = np.load('train_data.npy')
x1 = data_uniform[0,:]
t1 = data_uniform[1,:]

x1_sort=np.sort(x1)

""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """
intervals=np.array_split(x1_sort,M)
mu=np.zeros((1,M))
for i in range(M):
    mu[0,i]=np.median(intervals[i][:])
center_values=np.asarray(mu/(mu+1))
phi1=np.zeros((M,20))
for i in range(M):
    phi1[i,:]=np.exp((-(x1-mu[0,i])**2)/(2*s**2))
#phi1[:,0]=1;
w1=np.linalg.inv(phi1@phi1.T+ (0.1*np.identity(M)))@phi1@t1.T
mu1 = np.random.choice(x1_sort, size=M, replace=False)
phi3=np.zeros((M,20))
for i in range(M):
    phi3[i,:]=np.exp((-(x1-mu1[i])**2)/(2*s**2))

w2=np.linalg.inv(phi3@phi3.T+ (0.1*np.identity(M)))@phi3@t1.T

""" ======================== Load Test Data  and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
data_uniform = np.load('test_data.npy')
x2 = data_uniform.T
x2=np.linspace(min(x2), max(x2),10000)
Truefunction=x2/(x2+1)
phi2=np.zeros((M,10000))
for i in range(M):
    phi2[i,:]=np.exp((-(x2-mu[0,i])**2)/(2*s**2))
y=w1@phi2
phi4=np.zeros((M,10000))
for i in range(M):
    phi4[i,:]=np.exp((-(x2-mu1[i])**2)/(2*s**2))
y1=w2@phi4

""" ======================== prediction plots =========================== """


plt.figure(1)
plt.plot(x1,t1,"o",label="Training Data")
plt.plot(mu[0,:],center_values[0,:],"x",label="Center Values")
plt.ylim([-10,10])
plt.plot(x2,Truefunction,label="True Function")
plt.plot(x2,y,label="RBF center values")
plt.plot(x2,y1,label="RBF random values")
plt.ylabel("t")
plt.xlabel("x")
plt.title("Prediction of the data")
plt.legend()
plt.savefig('prediction.JPG')





""" ======================== for loop for evenly spaced error =========================== """
""" ======================  Variable Declaration ========================== """

abs_error_even_spaced=np.array([])

for M in range(20):
    M=M+1    
    """ =======================  Load Training Data ======================= """
    data_uniform = np.load('train_data.npy')
    x1 = data_uniform[0,:]
    t1 = data_uniform[1,:]
    """ ========================  Train the Model ============================= """
    """This is where you call functions to train your model with different RBF kernels   """
    intervals=np.array_split(x1_sort,M)
    mu=np.zeros((1,M))
    for i in range(M):
        mu[0,i]=np.median(intervals[i][:])
    phi1=np.zeros((M,20))
    for i in range(M):
        phi1[i,:]=np.exp((-(x1-mu[0,i])**2)/(2*s**2))
    w1=np.linalg.inv(phi1@phi1.T+ (0.1*np.identity(M)))@phi1@t1.T
    mu1 = np.random.choice(x1_sort, size=M, replace=False)
    phi3=np.zeros((M,20))
    for i in range(M):
        phi3[i,:]=np.exp((-(x1-mu1[i])**2)/(2*s**2))
    w2=np.linalg.inv(phi3@phi3.T+ (0.1*np.identity(M)))@phi3@t1.T
    """ ======================== Load Test Data  and Test the Model =========================== """
    
    """This is where you should load the testing data set. You shoud NOT re-train the model   """
    data_uniform = np.load('test_data.npy')
    x2 = data_uniform.T
    x2=np.linspace(min(x2), max(x2),10000)
    Truefunction=x2/(x2+1)
    phi2=np.zeros((M,10000))
    for i in range(M):
        phi2[i,:]=np.exp((-(x2-mu[0,i])**2)/(2*s**2))
    y=w1@phi2
    phi4=np.zeros((M,10000))
    for i in range(M):
        phi4[i,:]=np.exp((-(x2-mu1[i])**2)/(2*s**2))
    y1=w2@phi4
    abs_error_even_spaced=np.append(abs_error_even_spaced,np.sum(abs(y-Truefunction))/10000)
    
    
""" ========================  for loop for the avg error varying M  =========================== """
""" ======================  Variable Declaration ========================== """

abs_error_avg=np.array([])


for M in range(20):
    M=M+1 
    for k in range(10):
        """ =======================  Load Training Data ======================= """
        data_uniform = np.load('train_data.npy')
        x1 = data_uniform[0,:]
        t1 = data_uniform[1,:]
        """ ========================  Train the Model ============================= """
        """This is where you call functions to train your model with different RBF kernels   """
        intervals=np.array_split(x1_sort,M)
        mu=np.zeros((1,M))
        for i in range(M):
            mu[0,i]=np.median(intervals[i][:])
        phi1=np.zeros((M,20))
        for i in range(M):
            phi1[i,:]=np.exp((-(x1-mu[0,i])**2)/(2*s**2))
        w1=np.linalg.inv(phi1@phi1.T+ (0.1*np.identity(M)))@phi1@t1.T
        mu1 = np.random.choice(x1_sort, size=M, replace=False)
        phi3=np.zeros((M,20))
        for i in range(M):
            phi3[i,:]=np.exp((-(x1-mu1[i])**2)/(2*s**2))
        w2=np.linalg.inv(phi3@phi3.T+ (0.1*np.identity(M)))@phi3@t1.T
        """ ======================== Load Test Data  and Test the Model =========================== """
        
        """This is where you should load the testing data set. You shoud NOT re-train the model   """
        data_uniform = np.load('test_data.npy')
        x2 = data_uniform.T
        x2=np.linspace(min(x2), max(x2),10000)
        Truefunction=x2/(x2+1)
        phi2=np.zeros((M,10000))
        for i in range(M):
            phi2[i,:]=np.exp((-(x2-mu[0,i])**2)/(2*s**2))
        y=w1@phi2
        phi4=np.zeros((M,10000))
        for i in range(M):
            phi4[i,:]=np.exp((-(x2-mu1[i])**2)/(2*s**2))
        y1=w2@phi4
        abs_error_avg=np.append(abs_error_avg,np.sum(abs(y1-Truefunction))/10000)

abs_error_avg=np.reshape(abs_error_avg,(20,10))

means=np.array([])
for i in range(len(abs_error_avg)):
    means=np.append(means,np.mean(abs_error_avg[i]))
stds=np.array([])
for i in range(len(abs_error_avg)):
    stds=np.append(stds,np.std(abs_error_avg[i]))

CTEs = means
error = stds
x_pos = np.arange(1,21)










""" ======================== error plots =========================== """
fig, ax = plt.subplots()
ax.errorbar(x_pos, CTEs,
       yerr=stds,
       fmt='-o',
       ecolor='green',
       label="avg abs error random values")
ax.set_ylabel('mean of absolute error and standard deviation')
ax.set_xlabel('M')
ax.set_xticks(x_pos)
ax.set_xticklabels(np.arange(1,21))
ax.set_title('absolute error')
ax.yaxis.grid(True)
ax.plot(range(1,21),abs_error_even_spaced,"red", label="abs error center values")
#plt.plot(range(1,21),error_poly, "orange",label="Polynomial error")
ax.legend()
# Save the figure and show
plt.tight_layout()
plt.ylim([3,4])
plt.savefig('error1.JPG')

plt.show()




plt.figure(figsize =(10, 7))
plt.boxplot(abs_error_avg.T)
plt.plot(range(1,21),abs_error_even_spaced,"red", label="abs error center values")
#plt.plot(range(1,21),error_poly, "orange",label="Polynomial error")
plt.xlabel('M')
plt.ylabel('range of error')
plt.legend()
plt.ylim([3,4])
#plt.savefig('error1.JPG')
#I know its not labeled I just did it really quick the boxes are for the random errors



""" ======================== for loop for evenly spaced error =========================== """
""" ======================  Variable Declaration ========================== """

abs_error_even_spaced=np.array([])
number_of_s=20
M=5
for s in np.linspace(0.001,10,number_of_s):
 
    """ =======================  Load Training Data ======================= """
    data_uniform = np.load('train_data.npy')
    x1 = data_uniform[0,:]
    t1 = data_uniform[1,:]
    """ ========================  Train the Model ============================= """
    """This is where you call functions to train your model with different RBF kernels   """
    intervals=np.array_split(x1_sort,M)
    mu=np.zeros((1,M))
    for i in range(M):
        mu[0,i]=np.median(intervals[i][:])
    phi1=np.zeros((M,20))
    for i in range(M):
        phi1[i,:]=np.exp((-(x1-mu[0,i])**2)/(2*s**2))
    w1=np.linalg.inv(phi1@phi1.T+ (0.1*np.identity(M)))@phi1@t1.T
    mu1 = np.random.choice(x1_sort, size=M, replace=False)
    phi3=np.zeros((M,20))
    for i in range(M):
        phi3[i,:]=np.exp((-(x1-mu1[i])**2)/(2*s**2))
    w2=np.linalg.inv(phi3@phi3.T+ (0.1*np.identity(M)))@phi3@t1.T
    """ ======================== Load Test Data  and Test the Model =========================== """
    
    """This is where you should load the testing data set. You shoud NOT re-train the model   """
    data_uniform = np.load('test_data.npy')
    x2 = data_uniform.T
    x2=np.linspace(min(x2), max(x2),10000)
    Truefunction=x2/(x2+1)
    phi2=np.zeros((M,10000))
    for i in range(M):
        phi2[i,:]=np.exp((-(x2-mu[0,i])**2)/(2*s**2))
    y=w1@phi2
    phi4=np.zeros((M,10000))
    for i in range(M):
        phi4[i,:]=np.exp((-(x2-mu1[i])**2)/(2*s**2))
    y1=w2@phi4
    abs_error_even_spaced=np.append(abs_error_even_spaced,np.sum(abs(y-Truefunction))/10000)
    






""" ======================== for loop for the avg error varying s =========================== """
""" ======================  Variable Declaration ========================== """

abs_error_avg=np.array([])


for s in np.linspace(0.001,10,number_of_s):
    for k in range(10):
        """ =======================  Load Training Data ======================= """
        data_uniform = np.load('train_data.npy')
        x1 = data_uniform[0,:]
        t1 = data_uniform[1,:]
        """ ========================  Train the Model ============================= """
        """This is where you call functions to train your model with different RBF kernels   """
        intervals=np.array_split(x1_sort,M)
        mu=np.zeros((1,M))
        for i in range(M):
            mu[0,i]=np.median(intervals[i][:])
        phi1=np.zeros((M,20))
        for i in range(M):
            phi1[i,:]=np.exp((-(x1-mu[0,i])**2)/(2*s**2))
        w1=np.linalg.inv(phi1@phi1.T+ (0.1*np.identity(M)))@phi1@t1.T
        mu1 = np.random.choice(x1_sort, size=M, replace=False)
        phi3=np.zeros((M,20))
        for i in range(M):
            phi3[i,:]=np.exp((-(x1-mu1[i])**2)/(2*s**2))
        w2=np.linalg.inv(phi3@phi3.T+ (0.1*np.identity(M)))@phi3@t1.T
        """ ======================== Load Test Data  and Test the Model =========================== """
        
        """This is where you should load the testing data set. You shoud NOT re-train the model   """
        data_uniform = np.load('test_data.npy')
        x2 = data_uniform.T
        x2=np.linspace(min(x2), max(x2),10000)
        Truefunction=x2/(x2+1)
        phi2=np.zeros((M,10000))
        for i in range(M):
            phi2[i,:]=np.exp((-(x2-mu[0,i])**2)/(2*s**2))
        y=w1@phi2
        phi4=np.zeros((M,10000))
        for i in range(M):
            phi4[i,:]=np.exp((-(x2-mu1[i])**2)/(2*s**2))
        y1=w2@phi4
        abs_error_avg=np.append(abs_error_avg,np.sum(abs(y1-Truefunction))/10000)

abs_error_avg=np.reshape(abs_error_avg,(number_of_s,10))

means=np.array([])
for i in range(len(abs_error_avg)):
    means=np.append(means,np.mean(abs_error_avg[i]))
stds=np.array([])
for i in range(len(abs_error_avg)):
    stds=np.append(stds,np.std(abs_error_avg[i]))

CTEs = means
error = stds
x_pos = np.linspace(0.001,10,int(number_of_s))


fig, ax = plt.subplots()

ax.errorbar(x_pos, CTEs,
       yerr=stds,
       fmt='-o',
       ecolor='green',
       label="avg abs error random values")
ax.set_ylabel('mean of absolute error and standard deviation')
ax.set_xlabel('s')
ax.set_xticks(x_pos)
ax.set_xticklabels(np.linspace(0.001,10,int(number_of_s)))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_title('absolute error')
ax.plot(np.linspace(0.001,10,int(number_of_s)),abs_error_even_spaced,"red", label="abs error center values")
plt.ylim([3,4])
ax.yaxis.grid(True)

ax.legend()
# Save the figure and show
plt.tight_layout()

plt.savefig('error2.JPG')

plt.show()

""" ======================== Polynomial =========================== """


import matplotlib.pyplot as plt

def plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]):

    #plot everything
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') 

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
def fitdata(x,t,M):
	'''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''	
	#This needs to be filled in
	X = np.array([x**m for m in range(M+1)]).T
	w = np.linalg.inv(X.T@X+(0.1*np.identity(M+1)))@X.T@t
	return w
error_poly=np.array([])
for i in range(20):
    M=i;
    l = 0
    u = 1
    N = 10
    gVar = .1
    #data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T
    data_uniform=np.load('train_data.npy')
    x1 = data_uniform[0,:]
    t1 = data_uniform[1,:]
    data_uniform=np.load('test_data.npy')
    x2 = data_uniform  #get equally spaced points in the xrange
    x2=np.linspace(min(x2), max(x2),10000)
    t2 = x2/(x2+1)#compute the true function value
        
    fig = plt.figure()
    #plotData(x1, t1, x2, t2,legend=['Training Data', 'True Function'])
    
    
    
            
    
    w = fitdata(x1,t1,M)
    xrange = np.arange(l,u,0.001)  #get equally spaced points in the xrange
    X = np.array([x2**m for m in range(w.size)]).T
    esty = X@w #compute the predicted value
    error_poly=np.append(error_poly,np.mean(abs(esty-t2)))
    
    
    
    
#plotData(x1,t1,x2,t2,x2,esty,['Training Data', 'True Function', 'Estimated\nPolynomial'])
#plt.ylim([-10,10])
plt.figure(4)
plt.plot(error_poly, label="Polynomial error")
plt.title("mean of absolute error of the polynomial function")
plt.xlabel("M")
plt.ylabel("mean of absolute error")
plt.ylim([3,4])

plt.savefig('poly.JPG')

plt.show()
