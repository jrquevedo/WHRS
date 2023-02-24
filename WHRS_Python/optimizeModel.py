#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: quevedo
"""

import csv
from WHRS import WHRS
from WHRSOptimizer.WHRSskoptBayesian import WHRSskoptBayesian
import time
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Shared params
from expertsModel import getFileModel
fileModel=getFileModel()

outputPath='results/'
LoadRange=[60,100] # Modified Load range
LoadStep=5
LoadStepInterval=0.5 # numeric value in [0,0.5]. 
                     #If 0 no Load interval, 
                     #If 0.5 the intervals overlay all the values in LoadRange

def getLoadInterval(Load,LoadStep,LoadStepInterval):
    return [Load-LoadStep*LoadStepInterval,Load+LoadStep*LoadStepInterval]

# Bayesian Optimizer params
nIter=20
initRandP=10
OptimizerName='Bayesian'

# Influence variation
InfluVar=list(range(0,100+1,5))

# Util for save to csv
def saveCSV(f,rows):
    for row in rows:
        first=True
        for v in row:
            if first:
                first=False
            else:
                f.write(',')
            f.write('{}'.format(v))
        f.write('\n')


#%% Load model and create a WHRS Object

def loadModel(fileModel):
    with open(fileModel,'rt') as f:
        csvReader=csv.reader(f,delimiter=',',quoting=csv.QUOTE_MINIMAL)
        header=csvReader.__next__()
        for ih in range(len(header)):
            header[ih]=header[ih].split(' ')[0]
        W=csvReader.__next__()
        inlfu=csvReader.__next__()
        DifMM=csvReader.__next__()
    for ii in range(len(W)):
        W[ii]=float(W[ii])
        inlfu[ii]=float(inlfu[ii])
        DifMM[ii]=float(DifMM[ii])
    print('Model read from {}: '.format(fileModel),end='')
    for ii in range(len(W)):
        print('{:+7.4f}*{}'.format(W[ii],header[ii]),end='')
    print('')
    print('Influence:')
    for ii in range(len(inlfu)):
        print('  {:16}:{:+6.2f}%'.format(header[ii],inlfu[ii]*100))
    print('')    
    return [W,inlfu,DifMM,header]

def evalModel(W,Exergy_eff_WHRS,CO2_red,EPC):
    return Exergy_eff_WHRS*W[0]+CO2_red*W[1]+EPC*W[2]

[W,Influ,DifMM,header]=loadModel(fileModel)

WHRSObj=WHRS(PropsSIStore='memory')
WHRSObj.params_range['Load']=LoadRange

#%% Best for each variable
VW1s=[[1,0,0],[0,1,0],[0,0,-1]]
VNames1=['\Psi Ex','CO2\ reduction','EPC']
LoadMax1s=[]
for iv in range(len(VW1s)):
    W1=VW1s[iv]
    VName=VNames1[iv]
    print('Optimizing: {}({})'.format(VName,W1))
    LoadMax=[]
    for Load in range(LoadRange[0],LoadRange[1]+1,LoadStep):
        print('\nOptimize for Load={}'.format(Load))
        # WHRSObj=WHRS()
        WHRSObj.params_range['Load']=getLoadInterval(Load,LoadStep,LoadStepInterval)
        opt=WHRSskoptBayesian(WHRSObj,W1,nIter=nIter,initRandP=initRandP)
        vals=opt.maximize()
        print('Exergy_eff_WHRS={:7.4f}'.format(opt.Exergy_eff_WHRS))
        print('CO2_red        ={:7.4f}'.format(opt.CO2_red))
        print('EPC            ={:7.4f}'.format(opt.EPC))
        print('RankEvaluation ={:7.4f}'.format(opt.target))
        print('Opt. time      ={:5.2f}'.format(opt.getOptTime()))
        LoadMax.append(opt.getMaxValues())
    with open('{}Best_{}.csv'.format(outputPath,VName),'wt') as f:
        saveCSV(f,[opt.getMaxNames()])
        saveCSV(f,LoadMax)
    LoadMax1s.append(LoadMax)
    
#%% Plot Variable for its best and others' best
X=list(range(LoadRange[0],LoadRange[1]+1,LoadStep))
for ivr in range(len(VW1s)): # Variable to represent
    VName=VNames1[ivr]
    pos=8+ivr # Position in LoadMax
    plt.figure()
    plt.ylabel(r'${}$'.format(VName))
    for ivo in range(len(VW1s)): # Variable to optimize
        LoadMax=LoadMax1s[ivo]
        Y=[]
        for il in range(len(LoadMax)):
            Y.append(LoadMax[il][pos])
        plt.plot(X,Y,label=r'Optimizing ${}$'.format(VNames1[ivo]))
    plt.xlabel('Load')
    plt.legend()
    plt.savefig('{}{}_forLoadIntervalWhenOptimizingSelfAndOthers.pdf'.format(outputPath,VName))
    

#%% Global Best
WHRSObj=WHRS(PropsSIStore='memory')
WHRSObj.params_range['Load']=LoadRange
opt=WHRSskoptBayesian(WHRSObj,W,nIter=nIter,initRandP=initRandP)
opt.maximize()
print('Exergy_eff_WHRS={:7.4f}'.format(opt.Exergy_eff_WHRS))
print('CO2_red        ={:7.4f}'.format(opt.CO2_red))
print('EPC            ={:7.4f}'.format(opt.EPC))
print('RankEvaluation ={:7.4f}'.format(opt.target))
print('Opt. time      ={:5.2f}'.format(opt.getOptTime()))
GlobalMax=opt.getMaxValues()
names=opt.getMaxNames()
for i in range(len(GlobalMax)):
    print('{:15}={}'.format(names[i],GlobalMax[i]))
plt.figure()
opt.plot('{}{}_GlobalMax.pdf'.format(outputPath,OptimizerName))

#%% Change the Load using the Global Best
LoadGlobal=[]
for Load in range(LoadRange[0],LoadRange[1]+1,LoadStep):
    print('\nChange Load={}'.format(Load))
    # WHRSObj=WHRS()
    (Exergy_eff_WHRS,CO2_red,EPC)=WHRSObj(Load,GlobalMax[1],GlobalMax[2],GlobalMax[3],GlobalMax[4],GlobalMax[5],GlobalMax[6],GlobalMax[7])
    print('Exergy_eff_WHRS={:7.4f}'.format(Exergy_eff_WHRS))
    print('CO2_red        ={:7.4f}'.format(CO2_red))
    print('EPC            ={:7.4f}'.format(EPC))
    LoadGlobal.append([Load,GlobalMax[1],GlobalMax[2],GlobalMax[3],GlobalMax[4],GlobalMax[5],GlobalMax[7],GlobalMax[7],Exergy_eff_WHRS,CO2_red,EPC,evalModel(W,Exergy_eff_WHRS,CO2_red,EPC,),0])

#%% Best for each Load
LoadMax=[]
for Load in range(LoadRange[0],LoadRange[1]+1,LoadStep):
    print('\nOptimize for Load={}'.format(Load))
    # WHRSObj=WHRS()
    WHRSObj.params_range['Load']=getLoadInterval(Load,LoadStep,LoadStepInterval)
    opt=WHRSskoptBayesian(WHRSObj,W,nIter=nIter,initRandP=initRandP)
    vals=opt.maximize()
    print('Exergy_eff_WHRS={:7.4f}'.format(opt.Exergy_eff_WHRS))
    print('CO2_red        ={:7.4f}'.format(opt.CO2_red))
    print('EPC            ={:7.4f}'.format(opt.EPC))
    print('RankEvaluation ={:7.4f}'.format(opt.target))
    print('Opt. time      ={:5.2f}'.format(opt.getOptTime()))
    LoadMax.append(opt.getMaxValues())

#%% Save to csv

fcsvOLName='{}{}_OptimizeLoad.csv'.format(outputPath,OptimizerName)
with open(fcsvOLName,'wt') as fcsv:
    saveCSV(fcsv,[opt.getMaxNames(),GlobalMax])
    saveCSV(fcsv,LoadMax)
print('Writed optimizations'' output to {}'.format(fcsvOLName))

fcsvCLName='{}{}_ChangedLoad.csv'.format(outputPath,OptimizerName)
with open(fcsvCLName,'wt') as fcsv:
    saveCSV(fcsv,[opt.getMaxNames(),GlobalMax])
    saveCSV(fcsv,LoadGlobal)
print('Writed optimizations'' output to {}'.format(fcsvCLName))

#%% Plot optimize Load

def plotAxe(ax,X,LoadMax,col,MName,xlabel='Load'):
    Y=np.array(LoadMax)[:,col].tolist()
    ax.plot(X,Y)
    ax.set_ylabel(r'${}$'.format(MName))
    ax.set_xlabel(r'${}$'.format(xlabel))


X=list(range(LoadRange[0],LoadRange[1]+1,LoadStep))
Xtics=list(range(LoadRange[0],LoadRange[1]+1,5))

fig, axs = plt.subplots(2, 2,figsize=(8, 8),tight_layout=True)

for vax in axs:
    for ax in vax:
        ax.set_xticks(Xtics)

plotAxe(axs[0][0],X,LoadMax, 8,VNames1[0])
# plotAxe(axs[0][0],X,LoadGlobal, 8,'Exergy_eff_WHRS')
plotAxe(axs[1][0],X,LoadMax, 9,VNames1[1])
plotAxe(axs[0][1],X,LoadMax,10,VNames1[2])
plotAxe(axs[1][1],X,LoadMax,11,'Rank\ value')

fig.savefig('{}{}_Load.pdf'.format(outputPath,OptimizerName),dpi=300)

#%% Varing the influence

GlobalMaxInflu=[]
WI=abs(np.array(W.copy()))
WHRSObj.params_range['Load']=LoadRange
for EPCInfluP in InfluVar:
    if EPCInfluP==0:
        WI[2]=0
    elif EPCInfluP==100:
        WI[0]=0
        WI[1]=0
        WI[2]=-1
    else:
        EPCInflu=EPCInfluP/100
        WI[2]=-(WI[0]*DifMM[0]*EPCInflu+WI[1]*DifMM[1]*EPCInflu)/(DifMM[2]*EPCInflu-DifMM[2])
        WI[2]=np.sign(W[2])*WI[2]

    print('influence EPC:{:2}% -> WEPC:{:7.4f}'.format(EPCInfluP,WI[2]))
    
    # # Check
    # influ=np.array([abs(WI[0])*DifMM[0],abs(WI[1])*DifMM[1],abs(WI[2])*DifMM[2]])
    # influn=influ/sum(abs(influ))
    # print(WI)
    # print(influn)
    
    # Calculate the global Best
    opt=WHRSskoptBayesian(WHRSObj,WI,nIter=nIter,initRandP=initRandP)
    opt.maximize()
    print('EPC Influence  ={:6.2f}%%'.format(EPCInfluP))
    print('Exergy_eff_WHRS={:7.4f}'.format(opt.Exergy_eff_WHRS))
    print('CO2_red        ={:7.4f}'.format(opt.CO2_red))
    print('EPC            ={:7.4f}'.format(opt.EPC))
    print('RankEvaluation ={:7.4f}'.format(opt.target))
    print('Opt. time      ={:5.2f}'.format(opt.getOptTime()))
    GlobalMaxI=opt.getMaxValues()
    print(GlobalMaxI)    
    GlobalMaxInflu.append(GlobalMaxI)
    
#%% Plot the Load's Global Optimum from EPC Influence


fig, axs = plt.subplots(2, 2,figsize=(8, 8),tight_layout=True)

Xtics=list(range(0,100+1,10))
for vax in axs:
    for ax in vax:
        ax.set_xticks(Xtics)

X=InfluVar
plotAxe(axs[0][0],X,GlobalMaxInflu, 8,VNames1[0],'Influence of EPC (percent)')
plotAxe(axs[1][0],X,GlobalMaxInflu, 9,VNames1[1],'Influence of EPC (percent)')
plotAxe(axs[0][1],X,GlobalMaxInflu,10,VNames1[2],'Influence of EPC (percent)')
plotAxe(axs[1][1],X,GlobalMaxInflu,0,'Optimum\ Load','Influence of EPC (percent)')

fig.savefig('{}{}_Influence4.pdf'.format(outputPath,OptimizerName),dpi=300)

X=InfluVar
Y=[]
for i in range(len(GlobalMaxInflu)):
   Y.append(GlobalMaxInflu[i][0])
plt.figure()
plt.plot(X,Y)
plt.xlabel('Influence of EPC (percent)')
plt.ylabel('Optimum Load')
plt.xticks(X)
plt.yticks(list(range(60,100,5)))
plt.grid(axis='y')

# Plot the User preferences' point
plt.plot([Influ[2]],[GlobalMax[0]],'x')

plt.savefig('{}{}_Influence.pdf'.format(outputPath,OptimizerName))

plt.show()



