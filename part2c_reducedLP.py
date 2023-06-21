
# coding: utf-8

# In[41]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gurobipy as gp
import numpy as np
from numpy import matlib
from gurobipy import GRB
import scipy.sparse as sp
from itertools import compress
import time

time0 = time.time()
epstol = 1e-8

returns = np.loadtxt("data/full/returnsTrain_unstandardized.csv",delimiter=',')

reward = np.mean(returns,axis = 0) 
reward = np.loadtxt("data/full/muTrain.csv",delimiter=',')

risk = returns-reward
model = gp.Model('markowitzLP') # Creating model
t,n = returns.shape #t = days, n= num stocks 
print(t)
print(n)

mu = 100

w = model.addMVar(n,lb = 0) #adding weight variables
y = model.addMVar(t,lb = 0) #adding y variables  
coef1 = reward*mu #coefficients for weights 
coef2 = (np.matlib.repmat(-1/t,t,1)).ravel() #coefficients for y 
ones = (np.ones((n,1))).ravel() 
obj = w@coef1 + y@coef2 #objective function 
#A = sp.csr_matrix(risk)
A = risk
model.setObjective(obj, GRB.MAXIMIZE)
c1 = model.addConstr( -y<=A@w )
c2 = model.addConstr( y>=A@w )
c3 = model.addConstr( w.sum() <= 1 )
c4 = model.addConstr( -w.sum() <= -1 )

model.Params.method = 0 #primal simplex
model.optimize() 


rcs = w.RC
basis = w.VBasis
mu_coeffs = max(reward) - reward
var_coeffs = rcs + mu_coeffs*mu
nb_rcs = rcs[ basis != 0 ]
nb_mu_coeffs = mu_coeffs[ basis != 0 ]
nb_var_coeffs = var_coeffs[ basis != 0 ]
newmu = max(nb_var_coeffs/nb_mu_coeffs)
print(newmu)


portfolios = np.zeros((5000,n))
portfolios[0,:] = w.x
i = 1

# x, y, wm, wp, w+, w-
CM0 = np.vstack((-A,A,np.ones((1,n)),-np.ones((1,n))))
CM1 = np.vstack((-np.identity(t),-np.identity(t),np.zeros((2,t))))
CM2 = np.vstack((np.identity(t),np.zeros((t,t)),np.zeros((2,t))))
CM3 = np.vstack((np.zeros((t,t)),np.identity(t),np.zeros((2,t))))
CM4 = np.vstack((np.zeros((t,1)),np.zeros((t,1)),np.ones(1),np.zeros(1)))
CM5 = np.vstack((np.zeros((t,1)),np.zeros((t,1)),np.zeros(1),np.ones(1)))
ConsMat = np.hstack((CM0,CM2,CM3,CM4,CM5,CM1))
Dictb = np.concatenate((np.zeros(t), np.zeros(t), np.ones(1), -np.ones(1)))
Dictc = np.concatenate((np.zeros(n), np.zeros(2*t+2), coef2))
mu_coeffs = np.concatenate((reward, np.zeros(2*t+2), np.zeros(t)))
allcons = model.getAttr(GRB.Attr.CBasis, model.getConstrs())
basis = np.concatenate((w.VBasis, allcons, y.VBasis ))
cB = Dictc[basis == 0]
cN = Dictc[basis != 0]
mucB = mu_coeffs[basis == 0]
mucN = mu_coeffs[basis != 0]
DictBinv = np.linalg.inv(ConsMat[:,basis == 0])
DictN = ConsMat[:,basis != 0]
BinvN = DictBinv@DictN
NBVarC = np.transpose(BinvN)@cB - cN
NBMuC = np.transpose(BinvN)@mucB - mucN
BVal = DictBinv@Dictb
origi = np.arange(n+3*t+2)
Borig = origi[basis == 0]
Norig = origi[basis != 0]

sol = np.zeros(n+3*t+2)
posC = ( NBVarC <= epstol )
newmu = max(-NBVarC[posC]/NBMuC[posC])
print(newmu)

while (newmu > 0):
    newmuind = np.argwhere(((-NBVarC/NBMuC) == newmu) & (NBVarC <= epstol))
    tcol = BinvN[:,newmuind[0,0]]
    numer = BVal[Borig < n+2*t+2]
    denom = tcol[Borig < n+2*t+2]
    ratios = (numer[denom>epstol])/denom[denom>epstol]
    maxratio = min(ratios)
    maxratioind = np.argwhere((BVal/tcol == maxratio) & (tcol > epstol) & (Borig < n+2*t+2))

    
    basis[Borig[maxratioind[0,0]]] = 1
    basis[Norig[newmuind[0,0]]] = 0
    Borig = origi[basis==0]
    Norig = origi[basis!=0]
    cB = Dictc[basis == 0]
    cN = Dictc[basis != 0]
    mucB = mu_coeffs[basis == 0]
    mucN = mu_coeffs[basis != 0]
    DictBinv = np.linalg.inv(ConsMat[:,basis == 0])
    DictN = ConsMat[:,basis != 0]
    BinvN = DictBinv@DictN
    NBVarC = np.transpose(BinvN)@cB - cN
    NBMuC = np.transpose(BinvN)@mucB - mucN
    BVal = DictBinv@Dictb
    
    sol[basis == 0] = DictBinv@Dictb
    sol[basis != 0] = 0
    
    portfolios[i,:] = sol[0:n]
    i = i + 1

    poscoef = (NBVarC <= epstol)
    posmu = (NBMuC > epstol)
    posC = poscoef & posmu
    if any(posC):
        newmu = max(-NBVarC[posC]/NBMuC[posC])
    else:
        newmu = 0.0
    print(newmu)

np.count_nonzero(sum(portfolios))
class1 = sum(portfolios)>0

ypred = class1*1
np.savetxt("data/reduced_LP/ypred.csv", ypred, delimiter=",")


timefinal = time.time()

duration = timefinal-time0

# reducing data according to predictions 
sigma = np.loadtxt("data/full/sigmaTest.csv",delimiter=',')
mu = np.loadtxt("data/full/muTest.csv",delimiter=',')
with open('data/yfinance/tickers8year.csv') as f:
    tickers = f.read().splitlines()
    tickers = "".join(tickers).split(',')

muLP = mu[ypred == 1]
sigmaLP = sigma[ypred == 1]
sigmaLP = sigmaLP[:,ypred == 1]

tickersLP = list(compress(tickers, list(ypred==1)))

np.savetxt("data/reduced_LP/muTest.csv", muLP, delimiter=",")
np.savetxt("data/reduced_LP/sigmaTest.csv", sigmaLP, delimiter=",")
with open('data/reduced_LP/tickersLP.csv','w') as tfile:
	tfile.write(','.join(tickersLP))

