#from time import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import shutil
#from shutil import copyfile

import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model


#from keras.layers import Dense            ## not preferred
#from keras.models import Sequential       ## not preferred
#from keras.optimizers import Adam         ## not preferred
#from keras.models import model_from_json  ## not preferred
#from keras.utils import to_categorical    ## not preferred

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#### don't alter anything below this line
modelfile = 'b_nngmt'
mffp = os.path.join('b_saved_model', modelfile+'.h5')
if not os.path.isdir('b_saved_model'):
    os.mkdir('b_saved_model')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

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
    lenb=len(i.split('_b'))
    #print(lenb,len(i.split('_b')[0].split('label.')))
    if lenb==2 and len(i.split('_b')[0].split('label.'))==2:
                
                name0=i.split('_b')[0].split('label.')[1]
                name1=i.split('_b')[1]
                fl='label.'+name0+'_b'+name1
                fc='config.'+name0+'_b'+name1
                fl=os.path.join('data_train',fl)
                fc=os.path.join('data_train',fc)
                
                #fni[count]=i.split('.')[1]
                Xtrlist[count]=fc
                Ytrlist[count]=fl
                print(Xtrlist[count],Ytrlist[count])
                
                #arrX.append(np.load(os.path.join('data_train',Xtrlist[count])))
                #arrY.append(np.load(os.path.join('data_train',Ytrlist[count])))
                count=count+1

# automatically define dictionary labels 
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
labels2[cecdY]=labels[cdumy]
print('iy,Ydum[iy]:',iy,Ydum[iy])
print('encoded_Ydum[iy]:',ecdY)
print('tuple(dummy_y[iy]):',dumy)
print('encoded_Y[iy],labels2[encoded_Ydum[iy]]=',encoded_Ydum[iy],labels2[encoded_Ydum[iy]])      
print('cecdY=complement to encoded_Y[iy]=',cecdY)
print('labels2[cecdY]=',labels2[cecdY])      
print('')
# end automatically define dictionary labels 

dataX={};dataY={};
for ic in range(len(Ytrlist)):
    print('')
    print('***********************')
    #dataX[ic] = np.concatenate(arrX[:ic+1])
    #dataY[ic] = np.concatenate(arrY[:ic+1])
    #print('ic,dataX[i].shape',ic,dataX[ic].shape)
    #print('ic,dataY[i].shape',ic,dataY[ic].shape)
    #print(ic,Xtrlist[ic],Ytrlist[ic])
    # load dataset
    Xtr=Xtrlist[ic]
    Ytr=Ytrlist[ic]
    Ytrlabel=Ytr.split('.npy')[0].split('label.')[1]
    
    ######
    if not os.path.isdir('b_record'):
        os.mkdir('b_record')
        index=0
        os.mkdir(os.path.join('b_record', str(index)))
    else:
        if os.listdir('b_record')==[]:
            index=0
        else:
            index=-1000
            for i in os.listdir('b_record'):
                try:
                    if isinstance(int(i),int):
                        index=max(index,int(i))
                except:
                    zero=0
            index=1+index
        os.mkdir(os.path.join('b_record', str(index)))
        print("created directory",os.path.join('b_record', str(index)))    
 
    print('ic:',ic,'Ytrlabel',Ytrlabel)
    print('load Xtr:',Xtr)
    print('load Ytr:',Ytr)
    X_train=np.load(Xtr)
    Y_train=np.load(Ytr)     
    
    print('load dataset X_train, Y_train for ic = ',ic,'with dimension', \
          X_train.shape,Y_train.shape)
        
    Y=np.load(Ytr,allow_pickle=True)
    X=np.load(Xtr,allow_pickle=True)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    #dummy_y = np_utils.to_categorical(encoded_Y)
    dummy_y = to_categorical(encoded_Y)
    dim=dummy_y[0].shape[0]
    nofcolumn=X.shape[1]
    
    
    #### define, build and compile the model
#    def baseline_model():
#    # create model
#        model =  Sequential()
#        model.add(Dense(8, input_dim=nofcolumn, activation='relu'))
#        model.add(Dense(dim, activation='sigmoid'))
#       # Compile model
#        lrorig=0.005
#        opt = Adam(learning_rate=lrorig)
#        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#        return model    
        
    def load_model(mffp):
    # loading model
        #model = model_from_json(open('model_architecture.json').read())
        #model.load_weights('model_weights.h5')
        
        lrlow=0.001
        opt = Adam(learning_rate=lrlow)
#        model.load_weights(mffp)
#        model.compile( \
#            loss='binary_crossentropy',  \
#            optimizer=opt,metrics=[tensorflow.keras.metrics.BinaryAccuracy()] \
#            )
       
        model=tensorflow.keras.models.load_model(mffp)
        opt = Adam(learning_rate=lrlow)
        model.compile( \
            loss='binary_crossentropy',  \
            optimizer=opt,metrics=[tensorflow.keras.metrics.BinaryAccuracy()] \
            )
        #model.compile(loss="mean_squared_error" , optimizer=opt, metrics=["mean_squared_error"])            

        print('load and compile',mffp,'with a learning rate:',lrlow)
        return model

    def build_model():
    # create model
    
        model = Sequential()
        model.add(Dense(60, input_dim=nofcolumn, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
		        
        #input_layer = Input(shape=X_train.shape[1])
        #dense_layer_1 = Dense(60, activation='relu')(input_layer)
        #Dropout(rate=0.2)
        #dense_layer_2 = Dense(10, activation='sigmoid')(dense_layer_1)
        #Dropout(rate=0.2)
        #output = Dense(1)(dense_layer_2)        
        #model = Model(inputs=input_layer, outputs=output)
        
        lrorig=0.005
        opt = Adam(learning_rate=lrorig)
        print('compile a new',mffp,'with a learning rate:',lrorig)
        model.compile( \
            loss='binary_crossentropy',  \
            #loss='categorical_crossentropy',  \
            optimizer=opt,metrics=[tensorflow.keras.metrics.BinaryAccuracy()] \
            )
        #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def save_model(model):
        # saving model
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y%H%M")
        mffpdate = os.path.join('b_record',str(index),'saved_model',modelfile + '_' + dt_string+'.h5')
        assert isinstance(mffp, object)
        print('backup a copy of',mffp,'in',os.path.join('b_record',str(index),'saved_model'),'as',mffpdate)
        os.mkdir(os.path.join('b_record',str(index),'saved_model'))
        try:
            shutil.copy(mffp, mffpdate)
        except:
            zero=0
        #json_model = model.to_json()
        #open('model_architecture.json', 'w').write(json_model)
        # saving weights
        #model.save_weights('model_weights.h5', overwrite=True)
        #model.save_weights(mffp, overwrite=True)
        
        model.save(mffp,overwrite=True)
    
    if os.path.isfile(mffp): ## load model file if exists 
        print(mffp,'exists')
        print('loading',mffp)
        model = load_model(mffp)    
    else:
        ## if no model file exists, create one
        print(mffp,'does not exist. To build a new one')
        model = build_model()

    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=seed)  
	#X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_Y, test_size=0.3, random_state=seed)  	
    #history=model.fit(X_train, Y_train, epochs=2, batch_size=16, verbose=1) ## TS: 2, 16,1
    history=model.fit(X_train, Y_train, epochs=20, batch_size=2, verbose=1) ## production: 200,5,1
    save_model(model)
    
    # predictions
    print(' ')
    print('begin prediction')
    model = load_model(mffp)
    predictions = model.predict(X_test, verbose=1)
    predictions=[ ( round(i[0],0),round(i[1],0) ) for i in predictions.tolist() ]
    #predictions = model.predict(X_test > 0.5).astype("int32")


    #print(predictions)
    # reverse encoding
    count=0;
    ntneg=0;ntpos=0;nfneg=0;nfpos=0;
    #print('predictions',predictions)
    #print('labels2:',labels2)
    #print('labels:',labels)
    for pred in predictions:
        #print(Y_test[count], \
        #      labels[tuple(Y_test[count])], \
        #      labels2[pred])
        if labels[tuple(Y_test[count])]=='T' and \
           labels[pred]=='T':   ## true positive 
            ntpos=ntpos+1
        if labels[tuple(Y_test[count])]=='F' and \
           labels[pred]=='F':   ## true negative 
            ntneg=ntneg+1
        if labels[tuple(Y_test[count])]=='F' and \
           labels[pred]=='T':   ## false positive
            nfpos=nfpos+1
        if labels[tuple(Y_test[count])]=='T' and \
           labels[pred]=='F':   ## false negative
            nfneg=nfneg+1
        
        count=count+1
    print(count,len(predictions))
    print('ntneg,ntpos,nfneg,nfpos:',ntneg,ntpos,nfneg,nfpos)
    val_accuracy = (ntneg+ntpos)/len(predictions)
    print('val_accuracy (in %):',val_accuracy*100)
    
    df=pd.DataFrame(history.history)
    df[['loss','binary_accuracy']].plot(style='o-',figsize=(8, 5),title='loss, binary_accuracy vs. epoch')
    plt.grid(True)
    df.to_csv(os.path.join('b_record',str(index),'history.csv'), index = False)
    pngfn1=os.path.join('b_record',str(index),'history.png')
    plt.savefig(pngfn1)
    thisfile=Path(__file__).stem+'.py'
    copyfile(thisfile,os.path.join('record',str(index),thisfile))
    plt.show()
    