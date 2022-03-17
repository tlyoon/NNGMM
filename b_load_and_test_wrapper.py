## For this script to work, assure (1) the directory data_test/ is placed in the same directory as this script, 
## (2) The directory data_test/ contains config.*_b.npy, label.*_b.npy files.
## (3) A host of directories 1/ 2/ 3/, ... is placed in the same directory as this script. Each of the numbered directories contains a saved_model/ subdirectory with a *.h5 file, saved_model/b_nngmt_*.h5. 
## (4) An example of the directory hierarchy is as follow:

##   b_record/ --
##               |-- b_load_and_test_wrappter.py
##               |-- data_test/ --
##                                |
##                                |-- label.484500_b_p41@c27.npy, config.484500_b_p41@c27.npy
##                                |-- label.484500_b_p42@c27.npy, config.484500_b_p42@c27.npy
##                                |-- label.484500_b_p44@c27.npy, config.484500_b_p44@c27.npy
##                                |-- ...
##               |-- 0/saved_model/b_nngmt_270620210109.h5
##               |-- 1/saved_model/b_nngmt_270620210149.h5
##               |-- 2/saved_model/b_nngmt_270620210228.h5
##               |-- ...

## This script checks the performance of the existing nngmt model b_nngmt.h5 (kept in 0/saved_model/, 1/saved_model/, ...) using the data files kept in data_test/ in the forms of 
## {config.*_b.npy, label.*_b.npy} files. 

## The performance of the 0/saved_model/b_nngmt.h5 model, 1/saved_model/b_nngmt.h5 model, ..., after the model.predict() process with each {config.*_b.npy, label.*_b.npy} data in data_test/ is recorded 
## and saved in the directories record_test_0/, record_test_1/, ... .
## When finished, the summary of the performance of all b_nngmt*.h5 models in the numbered directories is recorded in a sigle file b_test_convergence.csv.

from pathlib import Path
import pandas as pd
#from tensorflow.keras.layers import Input, Dense
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow.keras.models
import numpy as np
import os
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import Adam
#import shutil


# automatically define dictionary labels based on one of the Y_test data file 
listdir=os.listdir('data_test')
count=0
Xtrlist={};Ytrlist={}
fni={};
arrY = [];arrX = []
#print('listdir:',listdir)
chosenlabelfile=[ i for i in listdir if len(i.split('.npy'))==2 and len(i.split('_b'))==2 and i.split('.')[0]=='label'][0]
#chosenlabelfile=os.path.join('data_test', chosenlabelfile)
#listdir=[chosenlabelfile]
#print('listdir:',listdir)
#for i in listdir:
i=chosenlabelfile
print('i',i)
lenb=len(i.split('_b'))
print('lenb',lenb)
if lenb==2 and len(i.split('_b')[0].split('label.'))==2:            
            name0=i.split('_b')[0].split('label.')[1]
            name1=i.split('_b')[1]
            fl='label.'+name0+'_b'+name1
            fc='config.'+name0+'_b'+name1
            fl=os.path.join('data_test',fl)
            fc=os.path.join('data_test',fc)
            Xtrlist[count]=fc
            Ytrlist[count]=fl
            print('Use the following sample test data to generate the labels dictionary')
            print(Xtrlist[count],Ytrlist[count])                
            count=count+1
# automatically define dictionary labels 
print('os.listdir()',os.listdir())
print('Ytrlist[0]:',Ytrlist[0])
print(os.path.isfile(Ytrlist[0]))
Ydum=np.load(Ytrlist[0],allow_pickle=True)
encoder = LabelEncoder()
encoder.fit(Ydum)
encoded_Ydum = encoder.transform(Ydum)
dummy_y = to_categorical(encoded_Ydum)
labels={};labels2={}
iy=0
st0=Ydum[iy]
ecdY=encoded_Ydum[iy]
dumy=tuple(dummy_y[iy])
labels[dumy]=st0
labels2[ecdY]=st0
fstate=['F','T']
fstate.remove(st0)
cst0=fstate[0]
cdumy=tuple([ -i + 1 for i in dummy_y[iy] ])
labels[cdumy]=cst0
cecdY=-encoded_Ydum[iy]+1
#labels2[cecdY]=labels[cdumy]
#print('iy,Ydum[iy]:',iy,Ydum[iy])
#print('encoded_Ydum[iy]:',ecdY)
#print('tuple(dummy_y[iy]):',dumy)
#print('encoded_Y[iy],labels2[encoded_Ydum[iy]]=',encoded_Ydum[iy],labels2[encoded_Ydum[iy]])      
#print('cecdY=complement to encoded_Y[iy]=',cecdY)
#print('labels2[cecdY]=',labels2[cecdY])      
print('labels=',labels)     
print('auto definition of dictionary lables completed') 
print('')
# end automatically define dictionary labels based on one of the Y_test data file 


try:
    os.remove('b_test_convergence.csv')
except:
    zero=0

cwd=os.getcwd()
#print('cwd:',cwd)
idir=[]
for i in os.listdir(cwd):
    try:
        inidx=int(i)
        idir.append(inidx)
    except:
       zero=0
idir.sort()

header='run_no. ' + 'b_nnmodel ' +  'name_test_data ' + 'no._of_test_data '  + 'accuracy '

ff=open('b_test_convergence.csv','w+')
ff.write(header)
ff.close()            

#print('idir:',idir)
print('')
iddarr=[]
Tleny={};avmale={};avmape={};avmse={};avRsq={}
for idd in idir:
    try:
        ff=open('b_test_convergence.csv','a+')
#        print('')
        h5dir=os.path.join(str(idd),'saved_model')
        h5path=os.listdir(h5dir)[-1]
        fullpath=os.path.join(h5dir,h5path)
        mffp = fullpath
        #print(mffp,os.path.isfile(mffp))
        
        #modelfile = 'nngmt'
        #mffp = os.path.join('saved_model', modelfile+'.h5')
        #print(mffp)
        
        mse_testlist={}
        Xtei={};Ytei={}
        Xte={};Yte={}
        countt=0
  
        for j in os.listdir('data_test'):
            #print('j:',j)
            if j.split('.')[0]=='config' and os.path.isfile(os.path.join('data_test', j)):
                fnt0=j.split('config.')[-1].split('.npy')[0]
                if '_b_' in fnt0:
                    fnt=j.split('config.')[-1].split('.npy')[0]
                    Xtei[countt]='config.'+fnt+'.npy'
                    Ytei[countt]='label.'+fnt+'.npy'
                    Xte[countt]=Xtei[countt]
                    Yte[countt]=Ytei[countt]
                    #print('countt,Xte:',Xte[countt],Xte[countt])    
                    #print('countt,Yte:',Yte[countt])  
                    countt=countt+1
                    #print('')
        #print('all test data:', [ [ Xte[i], Yte[i] ] for i in range(countt) ])
        #print('')
        TTleny=[];aavmale=[];aavmape=[];aavmse=[];aavRsq=[];countidd=0
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
            print('reloading the saved model',mffp,'to perform evaluation.')
            #model=tensorflow.keras.models.load_model(mffp)
            
            #from tensorflow.keras.optimizers import Adam
            #def load_model(mffp):
            # loading model
                #model = model_from_json(open('model_architecture.json').read())
                #model.load_weights('model_weights.h5')                
            #    lrlow=0.001
            #    opt = Adam(learning_rate=lrlow)               
            #    model=tensorflow.keras.models.load_model(mffp)
            #    opt = Adam(learning_rate=lrlow)
            #    model.compile( \
            #        loss='binary_crossentropy',  \
            #        optimizer=opt,metrics=[tensorflow.keras.metrics.BinaryAccuracy()] \
            #        )
            #    print('load and compile',mffp,'with a learning rate:',lrlow)
            #    return model
            
            #model=load_model(mffp)
            model=tensorflow.keras.models.load_model(mffp)
            #print('see me?')
        ### end load saved model #################    
            
        
        #### begins evaluations
        #    fmse_testlist=os.path.join('record',str(index),"metrics.dat")
        #    fmse = open(fmse_testlist, "w")      
            
            ### predict y_test using Y_test
            noconfigXte=X_test.shape[0]
            print('begin predicting testing data set',Xtei,Ytei)
            y_test = model.predict(X_test)
            
            ##
            predictions=[ ( round(i[0],0),round(i[1],0) ) for i in y_test.tolist() ]
            #print(predictions)
            # reverse encoding
            count=0;
            ntneg=0;ntpos=0;nfneg=0;nfpos=0;
            #print('predictions',predictions)
            #print('labels:',labels)
            Y_testarr=[];predarr=[]
            for pred in predictions:
#                print('Y_test[count]',Y_test[count])
#                print('labels[pred]:',labels[pred])
                if Y_test[count]=='T' and \
                   labels[pred]=='T':   ## true positive 
                    ntpos=ntpos+1
                if Y_test[count]=='F' and \
                   labels[pred]=='F':   ## true negative 
                    ntneg=ntneg+1
                if Y_test[count]=='F' and \
                   labels[pred]=='T':   ## false positive
                    nfpos=nfpos+1
                if Y_test[count]=='T' and \
                   labels[pred]=='F':   ## false negative
                    nfneg=nfneg+1
                
                count=count+1
                #predarr.append(labels[pred])
                #Y_testarr.append(labels[pred])
            print(count,len(predictions))
            print('ntneg,ntpos,nfneg,nfpos:',ntneg,ntpos,nfneg,nfpos)
            val_accuracy = (ntneg+ntpos)/len(predictions)
            print('val_accuracy (in %):',val_accuracy*100)         
            #header='run_no. ' + 'b_nnmodel ' +  'name_test_data ' + 'no._of_test_data '  + 'accuracy '
            ff=open('b_test_convergence.csv','a+')
            ff.write("{0:4d},{1:s},{2:s},{3:4d},{4:4.2f}".format(idd,h5path,Yte[n],len(Y_test),val_accuracy*100))
            ff.close()            

            
            ##
            ###mse_test=mean_squared_error(y_test, Y_test)
            ####print('mse_test:',mse_test)   
            ###mape_test=100*sum([ np.abs(Y_test[i] - y_test[i]) / np.abs(Y_test[i]) for i in range(len(y_test)) ])/len(y_test)
            #y_test_mean=np.mean(y_testloaded)
            ###y_test_mean=np.mean(y_test)
        #    sstot_test=np.sum([ (i[0]-y_test_mean)**2  for i in y_testloaded ])
            ###sstot_test=np.sum([ (i[0]-y_test_mean)**2  for i in y_test ])
            ###ssres_test=np.sum([(y_test[i][0]-Y_test[i])**2  for i in range(len(y_test)) ])
            #ssres_test=np.sum([(y_testloaded[i][0]-Y_test[i])**2  for i in range(len(y_testloaded)) ])
            ###Rsq_test=1-(ssres_test/sstot_test)
            ###male_te=sum([ np.log( np.abs((1+y_test[i][0])-(1+Y_test[i]))) for i in range(len(y_test))])/len(y_test)
            
        #### end evaluations
        
        
        ### subplots    
            ### train
        #    fig = plt.figure(figsize=(15,10))               
            #fig = plt.figure(figsize=(15,15))
            ###fig , ax = plt.subplots(nrows = 5, ncols = 1, figsize=(6,15))
            ###ymax=max(max(y_test.tolist())[0],max(Y_test))
            ###ymin=min(min(y_test.tolist())[0],min(Y_test))
            ###Deltay=(ymax-ymin)/100
            ###xlinear=np.arange(ymin,ymax,Deltay)
            ###ylinear=xlinear
            
            ###xrange=range(len(y_test))
            ###ix=0;iy=0;
            ###ax[ix].set_ylim(ymin, ymax)
            ###ax[ix].set_title("Y_test. No. of data: "+str(len(y_test)))
            #plt.scatter(xrange,Y_test, label="Y_test",color='blue')
            ###ax[ix].scatter(xrange,Y_test, label="Y_test",color='blue',s=1)
            
            #print('mean squared error of prediction using test dataset:',round(mse_test,2))
        
        ## subplot524 ax[1][1]
            #sub524=fig.add_subplot(5,2,4)
            ###ix=1;iy=0;
            ###ax[ix].set_ylim(ymin, ymax)
            #plt.title("y_test")
            ###ax[ix].set_title("y_test. No. of data: "+str(len(y_test)))
            #plt.scatter(xrange,y_test[ic], label="y_test",color="orange")
            ###ax[ix].scatter(xrange,y_test, label="y_test",color="orange",s=1)
            
            ## subplot526    ax[2][1]
            ###ix=3;iy=0;
            ###Y_tedict={}
            ###for i in range(len(Y_test)):
                ###Y_tedict[i]=Y_test[i]   
            ###Y_tedictsorted = np.array(sorted(Y_tedict.items(), key=lambda x: x[1]))
            #Y_trsorted=np.array(Y_trsorted)
            ###Y_tesorted=Y_tedictsorted[:,1]
            ###iY_tedict=[ int(i) for i in Y_tedictsorted[:,0]]
            ###y_tesorted=[ y_test[i] for i in iY_tedict ]
            ###xrange=range(len(Y_test))
            #sub526 = fig.add_subplot(5,2,6)
            ###ax[ix].set_ylim(ymin, ymax)
            ###ax[ix].set_title("Y_test, y_test sorted")
            ###ax[ix].scatter(range(len(y_tesorted)),y_tesorted,alpha=0.05, s=1,label="y_testsorted",color='orange')
            ###ax[ix].scatter(range(len(Y_tesorted)),Y_tesorted,alpha=0.05, s=1,label="Y_testsorted",color='blue')
            ###ax[ix].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
            ###ax[ix].set_xlabel("data index (sorted)")
            
        ## subplot528 ax[1][3] ax[3][1]
            #sub528=fig.add_subplot(5,2,8)
            ###ix=2;iy=0;
            ###ax[ix].set_ylim(ymin, ymax)
            ###ax[ix].scatter(xrange,Y_test, s=1, label="Y_test",color="blue")
            ###ax[ix].scatter(xrange,y_test, s=1, alpha=0.1, label="y_test",color="orange")
            ###ax[ix].set_title("y_test overlaid on Y_test")
            ###ax[ix].legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')
        
        ## sub52,10 ax[4][1]  
            ###ix=4;iy=0;
            ###ax[ix].set_xlim(ymin, ymax)
            ###ax[ix].set_ylim(ymin, ymax)
            ###state='No. of data: ' + str(len(y_test)) + '\n' + \
            ###"y_test vs. Y_test; mse: " + str(round(mse_test,2)) + '\n' + \
            ###'mape (in %): ' + str(round(mape_test[0],2)) + '\n' + \
            ###'R-squared: ' + str(round(Rsq_test,2)) + '\n' + \
            ###'male_te: '+ str(round(male_te,2))
            ###ax[ix].set_title(state)
            ###vax[ix].scatter(Y_test, y_test,s=1)
            ###ax[ix].plot(xlinear,ylinear)
            ###plt.tight_layout()
            ###if not os.path.isdir('record_test_'+str(idd)):
                ###os.mkdir('record_test_'+str(idd))

            ###pngfn=os.path.join('record_test_'+str(idd),'test.'+Xtefn+'.png')
            ###plt.savefig(pngfn)
            ###ff=open('b_test_convergence.csv','a')
            ###state=str(idd) +  ' ' + h5path + ' ' + Xtefn + ' ' + str(len(y_test)) + ' ' + str(round(mape_test[0],2)) +  ' ' + str(round(male_te,2)) + ' ' + str(round(mse_test,2)) + ' ' + str(round(Rsq_test,2)) + '\n'
            ###print('state',state)            
            ###ff.write(state)
			##
            ###TTleny.append(len(y_test))
            ###aavmape.append(mape_test[0])
            ###aavmale.append(male_te)
            ###aavmse.append(mse_test)
            ###aavRsq.append(Rsq_test)
            ###iddarr.append(idd)			
			##
            #plt.show()
            ###ff.close()
            ###countidd=countidd+1

        ###Tleny[idd]=np.sum(TTleny)
        ###avmape[idd]=np.sum(aavmape)/countidd
        ###avmale[idd]=np.sum(aavmale)/countidd
        ###avmse[idd]=np.sum(aavRsq)/countidd
        ###print('idd,Tleny[idd]:',idd,Tleny[idd])	
        ###print('idd,avmale[idd]:',idd,avmale[idd])					
    except:
       zero=0 

###ff=open('b_test_convergence.csv','a')
###ff.write('')
###ff.write('convergence trend: ')
###iddarr=list(set(iddarr))
###for i in iddarr:
    ###state='{:4d}'.format(i) + '{:8.3}'.format(Tleny[i]) + '{8.3}'.format(avmape[i]) \
	###+ '{8.3}'.format(avmale[i]) + '{8.3}'.format(avmse[i])
    ###ff.write(state)
###ff.close()
