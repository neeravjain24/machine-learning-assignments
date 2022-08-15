# -*- coding: utf-8 -*-
"""Assignment_5 (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A-p5kpnoxzuH4Tr44B_Q5W6rXcsFnaJB
"""

import numpy as np
import itertools
import math

X1 = np.arange(-10, 10, 0.25)
X2 = np.arange(-10, 10, 0.25)
X = np.array( list( itertools.product(X1, X2) ) )

def network_class(coordinate):
    x = coordinate[0]
    y = coordinate[1]
    if ( -9 <= x <= -7) and ( -5 <= y < 5):
        return [0,0]
    elif ( -7 <= x <= -3 ) and ( -5 <= y < -2):
        return [0,0]
    elif ( -3 <= x < -1 ) and ( -5 <= y < 5):
        return [0,0]
    elif ( 2 <= x <= 4 ) and ( -5 <= y < 5 ):
        return [1,1]
    elif ( 4 <= x < 6 ) and  ( -1 <= y < 1 ):
        return [1,1]
    elif ( 4 <= x <= 6 ) and ( 3 <= y < 5 ):
        return [1,1]
    elif ( 6 <= x < 8 ) and ( 3 <= y < 5 ):
        return [1,1]
    else:
        return [1,0]

y = []
for x in X:
    y.append( network_class(x) )

y = np.array(y)

def cell(x, w, b, activation):
    E = 0
    for i in range(len(x)):
        E += x[i]*w[i]
    E += b
    return (activation(E))

def cal_stp(y):
    if y>=0:
        return 1
    else:
        return 0
    
def linear(y):
    return y

def bnr_stp(y):
    if y>0:
        return 1
    else:
        return 0

def inverted(y):
    if y==0:
        return 1
    else:
        return 0


def predict(x, debug):
    
    y1_1 = cell(x,[1,0],0,linear)
    y1_2 = cell(x,[0,1],0,linear)
    
    
    x2 = [y1_1, y1_2]
    
    
    if debug :
        print('-------- layer 1 --------')
    y2_1 = cell(x2,[1,0],9,cal_stp)
    if debug :
        print(y2_1)
    y2_2 = cell(x2,[1,0],7,cal_stp)
    if debug :
        print(y2_2)
    y2_3 = cell(x2,[1,0],3,cal_stp)
    y2_4 = cell(x2,[1,0],1,cal_stp)
    
    y2_5 = cell(x2,[1,0],-2,cal_stp)
    y2_6 = cell(x2,[1,0],-4,cal_stp)
    y2_7 = cell(x2,[1,0],-6,cal_stp)
    y2_8 = cell(x2,[1,0],-8,cal_stp)
    
    y2_9 = cell(x2,[0,1],5,cal_stp)
    if debug :
        print(y2_9)
    y2_10 = cell(x2,[0,1],2,cal_stp)
    y2_11 = cell(x2,[0,1],1,cal_stp)
    y2_12 = cell(x2,[0,1],-1,cal_stp)
    y2_13 = cell(x2,[0,1],-3,cal_stp)
    y2_14 = cell(x2,[0,1],-5,cal_stp)
    if debug :
        print(y2_14)
    if debug :
        print('---------------------------')
    
    
    if debug :
        print('-------- layer 2 --------')
    y3_1 = cell([y2_1,y2_2],[1,-1],0,bnr_stp)
    if debug :
        print(y3_1)
    y3_2 = cell([y2_2,y2_3],[1,-1],0,bnr_stp)
    y3_3 = cell([y2_3,y2_4],[1,-1],0,bnr_stp)
    
    y3_4 = cell([y2_5,y2_6],[1,-1],0,bnr_stp)
    y3_5 = cell([y2_6,y2_7],[1,-1],0,bnr_stp)
    y3_6 = cell([y2_7,y2_8],[1,-1],0,bnr_stp)
    if debug :
        print('---------------------------')
    
    y3_7 = cell([y2_9,y2_14],[1,-1],0,bnr_stp)
    if debug :
        print(y3_7)
    y3_8 = cell([y2_9,y2_10],[1,-1],0,bnr_stp)
    y3_9 = cell([y2_11,y2_12],[1,-1],0,bnr_stp)
    y3_10 = cell([y2_13,y2_14],[1,-1],0,bnr_stp)
    if debug :
        print('---------------------------')
    
    
    if debug :
        print('-------- layer 3 --------')
    y4_1 = cell([y3_1,y3_7],[1,1],-1,bnr_stp)
    if debug :
        print(y4_1)
    y4_2 = cell([y3_2,y3_8],[1,1],-1,bnr_stp)
    y4_3 = cell([y3_3,y3_7],[1,1],-1,bnr_stp)
    
    y4_4 = cell([y3_4,y3_7],[1,1],-1,bnr_stp)
    y4_5 = cell([y3_5,y3_9],[1,1],-1,bnr_stp)
    y4_6 = cell([y3_5,y3_10],[1,1],-1,bnr_stp)
    y4_7 = cell([y3_6,y3_10],[1,1],-1,bnr_stp)
    if debug :
        print('---------------------------')
    
   
    if debug :
        print('-------- layer 4 --------')
    y5_1 = cell([y4_1,y4_2,y4_3],[1,1,1],0,bnr_stp)
    if debug :
        print(y5_1)
    y5_2 = cell([y4_4,y4_5,y4_6,y4_7],[1,1,1,1],0,bnr_stp)
    if debug :
        print('---------------------------')
    
    
    y6_1 = cell([y5_1,y5_2],[1,0],0,inverted)
    y6_2 = cell([y5_1,y5_2],[0,1],-1,inverted)
    
    return [y6_1, y6_2]

predictions = []
for x in X:
    predictions.append( predict(x, False) )

predictions = np.array(predictions)
predict([-9,5], True)

sum(predictions == y)

Ux = []
Uy = []
Fx = []
Fy = []
for i in range(len(X)):
    pred = predict(X[i], False)
    if pred == [0,0]:
        Ux.append(X[i][0])
        Uy.append(X[i][1])
    elif pred == [1,1]:
        Fx.append(X[i][0])
        Fy.append(X[i][1])

import matplotlib.pyplot as plt
plt.scatter(Ux,Uy)
plt.scatter(Fx,Fy)

