## Mandatory requirement 1: Existence of data_train/ containing at least 
## one pair of training data {label.XXX_c_YYY.npy, config.XXX_c_YYY.npy}. 

## Mandatory requirement 2: Existence of data_test/ containing {label.XXX_c_YYY.npy, config.XXX_c_YYY.npy} pairs which are not seen before by the trained model. 
## These data sets must not be the same as that used for training the model.

## If there is no existing *.h5 model file in saved_model/, the script will 
## create and train one, and then keep it in saved_model/, using the first-encountered training data 
## in data_training/. 
## 
## If an *.h5 model already existed in saved_model/, it will be re-trained
## using the first-encountered data files kept in data_train/. 
 
## The re-trained model is then used to predict the test data sets kept in data_test/. 

## The training performance of a retrained model after the model.fit() process
## with the current {config.*_c_*.npy, label.*_c_*.npy} training data is recorded  
## and saved in the numbered foldes record/0/, record/1/, record/2/, ... .
## The current trained model producing the records in any record/n/ directory is also saved in
## /record/n/saved_model
## The resulting output from this script are kept in the folders record/n/: 
## history.csv  history.png  itrain.py  metrics.dat  saved_model/nngmt_*.h5 train.*.png

### set mode ###
#mode='ts'    ### trouble shooting
mode='lo1'  # layer option 1
### end set mode

#from time import time
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
import shutil
from shutil import copyfile

#### don't alter anything below this line
modelfile = 'nngmt'
now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M")
mffp = os.path.join('saved_model', modelfile+'.h5')
mffpdate = os.path.join('saved_model', modelfile + '_' + dt_string+'.h5')
#mffp = os.path.join('saved_model', modelfile)
#mffpdate = os.path.join('saved_model', modelfile + '_' + dt_string)


###### create dict of all *_c*.npy data available in current dir
#listdir=os.listdir()
listdir=os.listdir('data_train')
count=0
Xtrlist={};Ytrlist={}
mse_testlist={}
#y_test={};
fni={};
arrY = [];arrX = []
for i in listdir:
    if len(i.split('.')) >=3 and i.split('.')[-1]=='npy' and \
        i.split('.')[0]=='config' and os.path.isfile(os.path.join('data_train','label.'+i.split('.')[1]+'.npy')            ):
            if not '' in i.partition('_c'):
                fni[count]=i.split('.')[1]
                Xtrlist[count]=i
                Ytrlist[count]='label.'+fni[count]+'.npy'
                print(Xtrlist[count],Ytrlist[count])
                arrX.append(np.load(os.path.join('data_train',Xtrlist[count])))
                arrY.append(np.load(os.path.join('data_train',Ytrlist[count])))
                count=count+1

dataX={};dataY={};
for ic in range(len(Ytrlist)):
    print('')
    print('***********************')
    dataX[ic] = np.concatenate(arrX[:ic+1])
    dataY[ic] = np.concatenate(arrY[:ic+1])
    print('ic,dataX[i].shape',ic,dataX[ic].shape)
    print('ic,dataY[i].shape',ic,dataY[ic].shape)
    #print(ic,Xtrlist[ic],Ytrlist[ic])
    # load dataset
    Xtr=Xtrlist[ic]
    Ytr=Ytrlist[ic]
    Ytrlabel=Ytr.split('.npy')[0].split('label.')[1]
    Xtr=os.path.join('data_train',Xtr)
    Ytr=os.path.join('data_train',Ytr)

    ######
    if not os.path.isdir('record'):
        os.mkdir('record')
        index=0
        os.mkdir(os.path.join('record', str(index)))
    else:
        if os.listdir('record')==[]:
            index=0
        else:
            index=-1000
            for i in os.listdir('record'):
                try:
                    if isinstance(int(i),int):
                        index=max(index,int(i))
                except:
                    zero=0
            index=1+index
        os.mkdir(os.path.join('record', str(index)))
        print("created directory",os.path.join('record', str(index)))    
 
    print('')
    Ytrlabel=str(ic)
    X_train=dataX[ic]
    Y_train=dataY[ic]
    print('load dataset dataX[ic], dataY[ic] for ic = ',ic,'with dimension', \
          dataX[ic].shape,dataY[ic].shape)    
    
    #### define, build and compile the model
    if ic==0:
        if mode=='ts':
            ### simple layers for TS
            input_layer = Input(shape=X_train.shape[1])
            dense_layer_1 = Dense(20, activation='relu')(input_layer)
#            Dropout(rate=0.05)
#            dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
            output = Dense(1)(dense_layer_1)
            ## simple layers for TS
    
        if mode=='lo1':
            ### layer option1
            input_layer = Input(shape=X_train.shape[1])
            dense_layer_1 = Dense(750, activation='relu')(input_layer)
            Dropout(rate=0.6)
            dense_layer_2 = Dense(100, activation='relu')(dense_layer_1)
            Dropout(rate=0.6)
            output = Dense(1)(dense_layer_2)
            ### end layer option1
    
    if os.path.isfile(mffp): ## load model file if exists
        lrlow=0.005
        print(mffp,'exists')
        #print('Making a dated copy of',mffp)
        #shutil.copy(mffp, mffpdate)
        print('loading',mffp)
        model=tensorflow.keras.models.load_model(mffp)
        opt = Adam(learning_rate=lrlow)
        print('compiling',mffp,'with a learning rate:',lrlow)
        model.compile(loss="mean_squared_error" , optimizer=opt, metrics=["mean_squared_error"])
    else:
        ## if no model file exists, create one
        lrorig=0.005
        print(mffp,'does not exist. To compile a new one with learning rate:',lrorig)
        opt = Adam(learning_rate=lrorig)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])
#### define, build and compile the model only once 
        

    ### fit the model 
    print('executing model.fit using training data',Xtr)
    if mode=='ts':
        history=model.fit(X_train, Y_train, batch_size=64, epochs=2, verbose=1, validation_split=0.2)
        print('model.fit done for ts mode')
    else:
        history=model.fit(X_train, Y_train, batch_size=8, epochs=6, verbose=1, validation_split=0.2)
        print('model.fit done for ts non-ts mode')
    ### end fit the model 
    
    #### save model after model.fit
    os.mkdir('saved_model') if not os.path.isdir('saved_model') else []
    #if os.path.isdir(mffp): 			### if model already existed
    if os.path.isfile(mffp): 			### if model already existed
        #os.remove(mffp) if os.path.isfile(mffp) else []
        print('saving',mffp,'after it is retrained with training data',Xtr)
        model.save(mffp)
        assert isinstance(mffp, object)
        print('saving a copy of',mffp,'in',os.path.join('record',str(index),'saved_model'))
        os.mkdir(os.path.join('record',str(index),'saved_model'))
        shutil.copy(mffp, os.path.join('record',str(index),mffpdate))
        #os.remove(mffpdate)   
     
    else:  ### if model does not exist
        model.save(mffp)
        assert isinstance(mffp, object)
        print('saving a brand-new copy of',mffp,'after it is trained with training data',Xtr)
            
    
    ### identify data files used to test th model from data_test/
    countt=0
    mse_testlist={}
    Xtei={};Ytei={}
    for j in os.listdir('data_test'):
        if j.split('.')[0]=='config' and os.path.isfile(os.path.join('data_test', j)):
            fnt0=j.split('config.')[-1].split('.npy')[0]
            if '_c_' in fnt0:
                fnt=j.split('config.')[-1].split('.npy')[0]
                Xtei[countt]='config.'+fnt+'.npy'
                Ytei[countt]='label.'+fnt+'.npy'
                countt=countt+1
    Xte=Xtei[0]
    Yte=Ytei[0]
    print('Xte:',Xte)    
    print('Yte:',Yte)    
    ### end identify data files used to test th model from data_test/
    
    ### identify data files used to test th model from data_test/
    #for j in os.listdir('data_test'):
    #    if j.split('.')[0]=='config':
    #        fnt=j.split('config.')[-1].split('.npy')[0]
    #Xte='config.'+fnt+'.npy'
    #Yte='label.'+fnt+'.npy'
    
    Xtefn=Xte.split('config.')[-1].split('.npy')[0]
    Xte=os.path.join('data_test',Xte)
    Yte=os.path.join('data_test',Yte)
    Ytelabel=Yte.split('.npy')[0].split('label.')[1]
    X_test = np.load(Xte)
    Y_test = np.load(Yte)
### end of identify data files used to test th model from data_test/

    
    
### check performance of current model against a common test dataset
### in data_test/

### load saved model #################
    print('reloading the saved model',mffp,'to perform evaluations that follow.')
    model=tensorflow.keras.models.load_model(mffp)
### end load saved model #################    
    

#### begins evaluations
    fmse_testlist=os.path.join('record',str(index),"metrics.dat")
    fmse = open(fmse_testlist, "w")  
    
    ### predict y_train using X_train
    noconfigXtr=X_train.shape[0]
    print('begin predicting full tranning data set containing',noconfigXtr,'configurations')
    y_train = model.predict(X_train)
    mse_train=mean_squared_error(y_train, Y_train)
    print('mse_train:',mse_train)
    mape_train=100*sum([ np.abs(Y_train[i] - y_train[i]) / np.abs(Y_train[i]) for i in range(len(y_train)) ])/len(y_train)
    y_train_mean=np.mean(y_train)
    sstot_train=np.sum([ (i-y_train_mean)**2  for i in y_train ])
    ssres_train=np.sum([(y_train[i]-Y_train[i])**2  for i in range(len(y_train)) ])
    Rsq_train=1-(ssres_train/sstot_train)   
    val_loss=round(history.history['val_loss'][-1],2)
    loss=round(history.history['loss'][-1],2)
    male_tr=sum([ np.log( np.abs((1+y_train[i][0])-(1+Y_train[i]))) for i in range(len(y_train))])/len(y_train)
    
    state='Total number of data sample in the training set: ' + str(noconfigXtr)
    fmse.write(state+'\n')
    state='loss during training: '+ str(round(loss,2))
    fmse.write(state+'\n')    
    state='val_loss during training: '+ str(round(val_loss,2))
    fmse.write(state+'\n')    
    state='mape (in %) for full training data: '+ str(round(mape_train[0],2))
    fmse.write(state+'\n')
    state='mean standard error for full training data: '+ str(round(mse_train,2))
    fmse.write(state+'\n')
    state='R-squared for full training data: '+ str(round(Rsq_train,2))
    fmse.write(state+'\n')
    state='mean absolute logarithmic error (male) for full training data: '+ str(round(male_tr,2))
    fmse.write(state+'\n')
    fmse.write('\n')
    
    
    ### predict y_test using Y_test
    noconfigXte=X_test.shape[0]
    print('begin predicting full testing data set',Xte)
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
    
    state='test set: ' + Xtefn
    fmse.write(state+'\n')
    state='mape (in %) for full testing data: '+ str(round(mape_test[0],2))
    fmse.write(state + '\n')
    state='mean standard error : '+ str(round(mse_test,2))
    fmse.write(state+'\n')
    state='R-square : '+ str(round(Rsq_test,2))
    fmse.write(state+'\n')
    state='mean absolute logarithmic error (male) for full testing data: '+ str(round(male_te,2))
    fmse.write(state+'\n')
    fmse.write('\n')
    state='last epoch : '+ str(len(history.history['loss']))
    fmse.write(state)
#### end evaluations


### subplots    
    ### train
#    fig = plt.figure(figsize=(15,10))               
    fig = plt.figure(figsize=(12,15))
    fig , ax = plt.subplots(nrows = 5, ncols = 2, figsize=(15,15))
    ymax=max(max(Y_train),max(y_train),max(y_test.tolist())[0],max(Y_test))
    ymin=min(min(Y_train),min(y_train),min(y_test.tolist())[0],min(Y_test))
    Deltay=(ymax-ymin)/100
    xlinear=np.arange(ymin,ymax,Deltay)
    ylinear=xlinear

## subplot00
    xrange=range(len(Y_train))
#    sub521 = fig.add_subplot(5,2,1)
#    sub521.set_ylim(ymin, ymax)
#    sub521.set_title("Y_train. No. of data: "+str(noconfigXtr))
#    sub521.scatter(xrange,Y_train,label="Y_train",color="blue",s=1)
    ax[0][0].set_ylim(ymin, ymax)
    ax[0][0].set_title("Y_train. No. of data: "+str(noconfigXtr))
    ax[0][0].scatter(xrange,Y_train,label="Y_train",color="blue",s=1)   
    
## subplot523
## subplot10
    xrange=range(len(y_train))
    #sub523 = fig.add_subplot(5,2,3)
    #sub523.set_ylim(ymin, ymax)
    #sub523.set_title('y_train. No. of data: '+str(noconfigXtr))
    #sub523.scatter(xrange,y_train, label="y_train",color="orange",s=1)
    ix=1;iy=0;
    ax[ix][iy].set_ylim(ymin, ymax)
    ax[ix][iy].set_title('y_train. No. of data: '+str(noconfigXtr))
    ax[ix][iy].scatter(xrange,y_train, label="y_train",color="orange",s=1)
    
    
## subplot525
## subplot20
    Y_trdict={}
    for i in range(len(Y_train)):
        Y_trdict[i]=Y_train[i]   
    Y_trdictsorted = np.array(sorted(Y_trdict.items(), key=lambda x: x[1]))
    #Y_trsorted=np.array(Y_trsorted)
    Y_trsorted=Y_trdictsorted[:,1]
    iY_trdict=[ int(i) for i in Y_trdictsorted[:,0]]
    y_trsorted=[ y_train[i] for i in iY_trdict ]
    xrange=range(len(Y_train))
    #sub525 = fig.add_subplot(5,2,5)
    #sub421.ylim([ymin,ymax])
    #sub525.set_ylim(ymin, ymax)
    #sub525.set_title("Y_train, y_train sorted")
    #sub525.scatter(range(len(y_trsorted)),y_trsorted,alpha=0.1, s=1,label="y_trainsorted",color='orange')
    #sub525.scatter(range(len(Y_trsorted)),Y_trsorted,s=1,label="Y_trainsorted",color='blue')
    #sub525.legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
    #sub525.set_xlabel("data index (sorted)")
    ix=3;iy=0;
    ax[ix][iy].set_ylim(ymin, ymax)
    ax[ix][iy].set_title("Y_train, y_train sorted")
    ax[ix][iy].scatter(range(len(y_trsorted)),y_trsorted,alpha=0.05, s=1,label="y_trainsorted",color='orange')
    ax[ix][iy].scatter(range(len(Y_trsorted)),Y_trsorted,alpha=0.05,s=1,label="Y_trainsorted",color='blue')
    ax[ix][iy].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
    ax[ix][iy].set_xlabel("data index (sorted)")


## subplot527
## ax[3][0]
    ix=2;iy=0;
    xrange=range(len(y_train))
#    sub527 = fig.add_subplot(5,2,7)
#    sub527.set_ylim(ymin, ymax)
#    sub527.set_title("y_train vs. Y_train")
#    plt.title("y_train overlaid on Y_train")
#    sub527.scatter(xrange,Y_train, s=1,label="Y_train",color="blue")
#    sub527.scatter(xrange,y_train,  s=1, alpha=0.1, label="y_train",color="orange")
#    sub527.legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
    xrange=range(len(y_train))
    ax[ix][iy].set_ylim(ymin, ymax)
    ax[ix][iy].set_title("y_train overlaid on Y_train")
    #ax[3][0].title("y_train overlaid on Y_train")
    ax[ix][iy].scatter(xrange,Y_train, s=1,label="Y_train",color="blue")
    ax[ix][iy].scatter(xrange,y_train,  s=1, alpha=0.1, label="y_train",color="orange")
    ax[ix][iy].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')


## subplot529
## ax[4][0]
    #sub529=fig.add_subplot(5,2,9)
    #sub529.set_xlim(ymin, ymax)
    #sub529.set_ylim(ymin, ymax)
    ax[4][0].set_xlim(ymin, ymax)
    ax[4][0].set_ylim(ymin, ymax)
    state="No. of data: " + str(noconfigXtr) + '\n' + \
    "y_train vs. Y_train; mse: " + str(round(mse_train,2))  + '\n'      + \
    'mape (in %): ' + str(round(mape_train[0],2)) + '\n'                + \
    'R-squared: '  + str(round(Rsq_train,2)) + '\n'                     + \
    'loss during training: '+ str(round(loss,2)) + '\n'                 + \
    'val_loss during training: ' + str(round(val_loss,2)) + '\n'        + \
    'male_tr: '+ str(round(male_tr,2))                    
    ax[4][0].set_title(state)
    ax[4][0].scatter(Y_train, y_train,label="y_train vs. Y_train",s=1)
    ax[4][0].plot(xlinear,ylinear)

     
    ### test
    xrange=range(len(y_test))

## subplot522  ax[0][1]
    #sub522=fig.add_subplot(5,2,2)
    ax[0][1].set_ylim(ymin, ymax)
    ax[0][1].set_title("Y_test. No. of data: "+str(len(y_test)))
    #plt.scatter(xrange,Y_test, label="Y_test",color='blue')
    ax[0][1].scatter(xrange,Y_test, label="Y_test",color='blue',s=1)
    
    #print('mean squared error of prediction using test dataset:',round(mse_test,2))

## subplot524 ax[1][1]
    #sub524=fig.add_subplot(5,2,4)
    ax[1][1].set_ylim(ymin, ymax)
    #plt.title("y_test")
    ax[1][1].set_title("y_test. No. of data: "+str(len(y_test)))
    #plt.scatter(xrange,y_test[ic], label="y_test",color="orange")
    ax[1][1].scatter(xrange,y_test, label="y_test",color="orange",s=1)
    
    ## subplot526    ax[2][1]
    ix=3;iy=1;
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
    ax[ix][iy].set_ylim(ymin, ymax)
    ax[ix][iy].set_title("Y_test, y_test sorted")
    ax[ix][iy].scatter(range(len(y_tesorted)),y_tesorted,alpha=0.05, s=1,label="y_testsorted",color='orange')
    ax[ix][iy].scatter(range(len(Y_tesorted)),Y_tesorted,alpha=0.05, s=1,label="Y_testsorted",color='blue')
    ax[ix][iy].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
    ax[ix][iy].set_xlabel("data index (sorted)")
    
## subplot528 ax[1][3] ax[3][1]
    #sub528=fig.add_subplot(5,2,8)
    ix=2;iy=1;
    ax[ix][iy].set_ylim(ymin, ymax)
    ax[ix][iy].scatter(xrange,Y_test, s=1, label="Y_test",color="blue")
    ax[ix][iy].scatter(xrange,y_test, s=1, alpha=0.1, label="y_test",color="orange")
    ax[ix][iy].set_title("y_test overlaid on Y_test")
    ax[ix][iy].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')

## sub52,10 ax[4][1]  
    #sub5210=fig.add_subplot(5,2,10)
    ax[4][1]  .set_xlim(ymin, ymax)
    ax[4][1]  .set_ylim(ymin, ymax)
    state='No. of data: ' + str(len(y_test)) + '\n' + \
    "y_test vs. Y_test; mse: " + str(round(mse_test,2)) + '\n' + \
    'mape (in %): ' + str(round(mape_test[0],2)) + '\n' + \
    'R-squared: ' + str(round(Rsq_test,2)) + '\n' + \
    'male_te: '+ str(round(male_te,2))
    ax[4][1]  .set_title(state)
    ax[4][1]  .scatter(Y_test, y_test,s=1)
    ax[4][1]  .plot(xlinear,ylinear)
    
    
    plt.tight_layout()
    #    plt.gca().set_aspect('equal')    
    pngfn=os.path.join('record',str(index),'train.'+str(ic)+'.test.'+Xtefn+'.png')
    plt.savefig(pngfn)
    
    df=pd.DataFrame(history.history)
    df[['loss','val_loss']].plot(style='o-',figsize=(8, 5),title='loss, val_loss vs. epoch')
    plt.grid(True)
    df.to_csv(os.path.join('record',str(index),'history.csv'), index = False)
    pngfn1=os.path.join('record',str(index),'history.png')
    plt.savefig(pngfn1)
    plt.show()
    fmse.close()

thisfile=Path(__file__).stem+'.py'
copyfile(thisfile,os.path.join('record',str(index),thisfile))
