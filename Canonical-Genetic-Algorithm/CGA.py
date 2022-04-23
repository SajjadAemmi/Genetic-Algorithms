from copy import deepcopy
import numpy as np
from numpy import linalg as LA
import pandas as pd
from matplotlib import pyplot as plt
import random
from math import *

def evaluate(P):

    meu, L = P.shape
    xmin = -10
    xmax = 10
    F = np.zeros((meu))

    for p in range(meu):
        m = 0
        for i in range(L):
            m = m * 2 + P[p, i]
        
        x = xmin + (xmax - xmin) * m / (2**L - 1)
        F[p] = abs(cos(x) * np.exp(-abs(x)/5))
    
    return F

#------------------------------------------------

def select(P, F):

    meu, L = P.shape
    parents = P.copy()
    S = np.sum(F)

    for p in range(meu):
        r = np.random.rand() * S
        a = 0
        i = -1
        while a < r:
            i = i + 1
            a = a + F[i]
        
        parents[p, :] = P[i, :]

    return parents

#------------------------------------------------

def crossover(P, Pc):
    
    meu, L = P.shape
    C = P.copy()

    for p in range(0, meu, 2):
        if np.random.rand() < Pc:
            i = int(np.random.rand() * L)
            C[p, 0:i] = P[p, 0:i]
            C[p+1, 0:i] = P[p+1, 0:i]
            C[p, i:L] = P[p+1, i:L]
            C[p+1, i:L] = P[p, i:L]
        else:
            C[p, :] = P[p, :]
            C[p+1, :] = P[p+1, :]

    return C

#------------------------------------------------

def mutate(childs, Pm):
    
    meu, L = childs.shape

    for p in range(meu):
        for i in range(L):
            if np.random.rand() < Pm:
                childs[p, i] = 1 - childs[p, i]

    return childs

#------------------------------------------------

meu = 6 #Number of people
L = 20 #Gene length

pm = 5/L #Probability of Mutate
pc = 0.7 #Probability of Crossover

max_gen = 100 #Maximum generation
FB = np.zeros(max_gen) #Best Fitness
FW = FB.copy() #Worst Fitness
FM = FB.copy() #Mean Fitness

for run in range(10):
    
    P = (np.random.rand(meu,L) > 0.5) * 1
    t = 0
    
    Fbest = np.zeros(max_gen)
    Fmean = np.zeros(max_gen)
    Fworst = np.zeros(max_gen)
    Pbest = np.zeros((max_gen,L))
    
    for t in range(max_gen):

        F = evaluate(P)
        Fbest[t] = np.max(F)
        p = np.argmax(F)
        Fbest[t] = np.max(Fbest)
        Fmean[t] = np.mean(F)
        Fworst[t] = np.min(F)
        Pbest[t,:] = P[p,:]
        
        parents = select(P,F)
        
        childs = crossover(parents,pc)
        
        childs = mutate(childs,pm)

        P = childs.copy()

    FB = FB + Fbest
    FM = FM + Fmean
    FW = FW + Fworst

print("FB")
print(FB)
print("FM")
print(FM)
print("FW")
print(FW)

plt.plot(FB, 'g', label = "Best")
plt.plot(FM, 'b', label = "Mean")
plt.plot(FW, 'r', label = "Worst")
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel('Generation')
plt.ylabel('Evaluation')
plt.show()