## For this script to work, assure the existence of (1) saved_model/nngmt.h5, and config.*_c.npy, label.*_c.npy files in data_test/ 

## This script uses the existing nngmt model nngmt.h5 (kept in saved_model/)
## using the data files kept in data_test/ in the forms of 
## unique pairs of legitimate {config.*_c.npy, label.*_c.npy} files. 

## If there exist more than one pair of legitimate 
## {config.*_c.npy, label.*_c.npy} files in data_test/, it should work as well.

## The training performance of the saved_model/nngmt.h5 model after the model.fit() process
## with each {config.*_c.npy, label.*_c.npy} data is recorded 
## and saved in the current directory.

from pathlib import Path
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow.keras.models
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.optimizers import Adam
#import shutil

#### don't alter anything below this line
modelfile = 'nngmt'
mffp = os.path.join('saved_model', modelfile+'.h5')

mse_testlist={}
Xtei={};Ytei={}
Xte={};Yte={}
countt=0
for j in os.listdir('data_test'):
    if j.split('.')[0]=='config' and os.path.isfile(os.path.join('data_test', j)):
        fnt0=j.split('config.')[-1].split('.npy')[0]
        if '_c_' in fnt0:
            fnt=j.split('config.')[-1].split('.npy')[0]
            Xtei[countt]='config.'+fnt+'.npy'
            Ytei[countt]='label.'+fnt+'.npy'
            Xte[countt]=Xtei[countt]
            Yte[countt]=Ytei[countt]
            #print('countt,Xte:',Xte[countt],Xte[countt])    
            #print('countt,Yte:',Yte[countt])  
            countt=countt+1
            print('')
print('all test data:', [ [ Xte[i], Yte[i] ] for i in range(countt) ])
print('')


for n in range(countt):    
    print('')
    Xtefn=Xte[n].split('.npy')[0].split('config.')[-1]
    #Xtefn=Xte[n].split('config.')[-1].split('.npy')[0]
    Xtei=os.path.join('data_test',Xte[n])
    Ytei=os.path.join('data_test',Yte[n])
    print('test data to evaluate:', Xtei, Ytei)        
    X_test = np.load(Xtei)
    Y_test = np.load(Ytei)
### end of identify data files used to test th model from data_test/
   
    
### check performance of current model against a common test dataset
### in data_test/

### load saved model #################
    print('reloading the saved model',mffp,'to perform evaluations that follow.')
    model=tensorflow.keras.models.load_model(mffp)
### end load saved model #################    
    

#### begins evaluations
#    fmse_testlist=os.path.join('record',str(index),"metrics.dat")
#    fmse = open(fmse_testlist, "w")      
    
    ### predict y_test using Y_test
    noconfigXte=X_test.shape[0]
    print('begin predicting testing data set',Xtei,Ytei)
    y_test = model.predict(X_test)
    mse_test=mean_squared_error(y_test, Y_test)
    print('mse_test:',mse_test)   
    mape_test=100*sum([ np.abs(Y_test[i] - y_test[i]) / np.abs(Y_test[i]) for i in range(len(y_test)) ])/len(y_test)
    #y_test_mean=np.mean(y_testloaded)
    y_test_mean=np.mean(y_test)
#    sstot_test=np.sum([ (i[0]-y_test_mean)**2  for i in y_testloaded ])
    sstot_test=np.sum([ (i[0]-y_test_mean)**2  for i in y_test ])
    ssres_test=np.sum([(y_test[i][0]-Y_test[i])**2  for i in range(len(y_test)) ])
    #ssres_test=np.sum([(y_testloaded[i][0]-Y_test[i])**2  for i in range(len(y_testloaded)) ])
    Rsq_test=1-(ssres_test/sstot_test)
    male_te=sum([ np.log( np.abs((1+y_test[i][0])-(1+Y_test[i]))) for i in range(len(y_test))])/len(y_test)
    
    #state='test set: ' + Xtefn
    #fmse.write(state+'\n')
    #state='mape (in %) for full testing data: '+ str(round(mape_test[0],2))
    #fmse.write(state + '\n')
    #state='mean standard error : '+ str(round(mse_test,2))
    #fmse.write(state+'\n')
    #state='R-square : '+ str(round(Rsq_test,2))
    #fmse.write(state+'\n')
    #state='mean absolute logarithmic error (male) for full testing data: '+ str(round(male_te,2))
    #fmse.write(state+'\n')
    #fmse.write('\n')
    #state='last epoch : '+ str(len(history.history['loss']))
    #fmse.write(state)
#### end evaluations


### subplots    
    ### train
#    fig = plt.figure(figsize=(15,10))               
    #fig = plt.figure(figsize=(15,15))
    fig , ax = plt.subplots(nrows = 5, ncols = 1, figsize=(6,15))
    ymax=max(max(y_test.tolist())[0],max(Y_test))
    ymin=min(min(y_test.tolist())[0],min(Y_test))
    Deltay=(ymax-ymin)/100
    xlinear=np.arange(ymin,ymax,Deltay)
    ylinear=xlinear
    
    xrange=range(len(y_test))
    ix=0;iy=0;
    ax[ix].set_ylim(ymin, ymax)
    ax[ix].set_title("Y_test. No. of data: "+str(len(y_test)))
    #plt.scatter(xrange,Y_test, label="Y_test",color='blue')
    ax[ix].scatter(xrange,Y_test, label="Y_test",color='blue',s=1)
    
    #print('mean squared error of prediction using test dataset:',round(mse_test,2))

## subplot524 ax[1][1]
    #sub524=fig.add_subplot(5,2,4)
    ix=1;iy=0;
    ax[ix].set_ylim(ymin, ymax)
    #plt.title("y_test")
    ax[ix].set_title("y_test. No. of data: "+str(len(y_test)))
    #plt.scatter(xrange,y_test[ic], label="y_test",color="orange")
    ax[ix].scatter(xrange,y_test, label="y_test",color="orange",s=1)
    
    ## subplot526    ax[2][1]
    ix=3;iy=0;
    Y_tedict={}
    for i in range(len(Y_test)):
        Y_tedict[i]=Y_test[i]   
    Y_tedictsorted = np.array(sorted(Y_tedict.items(), key=lambda x: x[1]))
    #Y_trsorted=np.array(Y_trsorted)
    Y_tesorted=Y_tedictsorted[:,1]
    iY_tedict=[ int(i) for i in Y_tedictsorted[:,0]]
    y_tesorted=[ y_test[i] for i in iY_tedict ]
    xrange=range(len(Y_test))
    #sub526 = fig.add_subplot(5,2,6)
    ax[ix].set_ylim(ymin, ymax)
    ax[ix].set_title("Y_test, y_test sorted")
    ax[ix].scatter(range(len(y_tesorted)),y_tesorted,alpha=0.05, s=1,label="y_testsorted",color='orange')
    ax[ix].scatter(range(len(Y_tesorted)),Y_tesorted,alpha=0.05, s=1,label="Y_testsorted",color='blue')
    ax[ix].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
    ax[ix].set_xlabel("data index (sorted)")
    
## subplot528 ax[1][3] ax[3][1]
    #sub528=fig.add_subplot(5,2,8)
    ix=2;iy=0;
    ax[ix].set_ylim(ymin, ymax)
    ax[ix].scatter(xrange,Y_test, s=1, label="Y_test",color="blue")
    ax[ix].scatter(xrange,y_test, s=1, alpha=0.1, label="y_test",color="orange")
    ax[ix].set_title("y_test overlaid on Y_test")
    ax[ix].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')

## sub52,10 ax[4][1]  
    ix=4;iy=0;
    ax[ix].set_xlim(ymin, ymax)
    ax[ix].set_ylim(ymin, ymax)
    state='No. of data: ' + str(len(y_test)) + '\n' + \
    "y_test vs. Y_test; mse: " + str(round(mse_test,2)) + '\n' + \
    'mape (in %): ' + str(round(mape_test[0],2)) + '\n' + \
    'R-squared: ' + str(round(Rsq_test,2)) + '\n' + \
    'male_te: '+ str(round(male_te,2))
    ax[ix].set_title(state)
    ax[ix].scatter(Y_test, y_test,s=1)
    ax[ix].plot(xlinear,ylinear)
    
    plt.tight_layout()
    if not os.path.isdir('record_test'):
        os.mkdir('record_test')
    pngfn=os.path.join('record_test','test.'+Xtefn+'.png')
    plt.savefig(pngfn)
    
    #plt.show()
#    fmse.close()
