### used in conjuction with ctrain.py
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from IPython.display import Image, display


#rdir='record_18_p1@cca'
#rdir='record_17_p9@c25'
#rdir='record_19_p1@c22'
rdir='record'
dirs=os.listdir(rdir)
dummy=[]
for i in dirs:
    try: 
        i=int(i)
        dummy.append(i)
    except:
        zero=0
dirs=dummy    
dirs=np.sort([ int(i) for i in dirs])    
    
val_loss={};loss={};xrange=[];anofconfig=0;
mape_tr={};mape_te={};Rsq_tr={};Rsq_te={};male_tr={};male_te={}
for i in dirs:
#        if os.path.isdir(os.path.join('record',i)) and isinstance(int(i), int):
        if os.path.isdir(os.path.join(rdir,str(i))):
            hf=os.path.join(rdir,str(i),'history.csv')
            metrics=os.path.join(rdir,str(i),'metrics.dat')
            if os.path.isfile(hf) and os.path.isfile(metrics):
                        f = open(metrics, "r")
                        data=f.readlines()
                        nofconfig=int(data[0].split(':')[-1].strip())
                        anofconfig=anofconfig+nofconfig
                        xrange.append(anofconfig)
                        #print(i,hf)
                        df = pd.read_csv(hf)
                        #loss=round(df['loss'].tolist()[-1],2)
                        #val_loss=round(df['val_loss'].tolist()[-1],2)
                        val_loss[anofconfig]=round(df['val_loss'].tolist()[-1],2)
                        loss[anofconfig]=round(df['loss'].tolist()[-1],2)
                        
                        mape_tr[anofconfig]=data[3].split(':')[-1].strip()
                        Rsq_tr[anofconfig]=data[5].split(':')[-1].strip();
                        male_tr[anofconfig]=data[6].split(':')[-1].strip();
                        
                        mape_te[anofconfig]=data[9].split(':')[-1].strip()
                        Rsq_te[anofconfig]=data[11].split(':')[-1].strip()
                        male_te[anofconfig]=data[12].split(':')[-1].strip()
                        
                        nofepoch=df.index.stop
#                        print(nofconfig,loss[nofconfig],val_loss[nofconfig])
xrange=np.sort(list(set(xrange)))
fig = plt.figure(figsize=(10,6))

subp = fig.add_subplot(221)
plt.title("loss, val_loss vs. number of data."+ '\n' + \
          " Number of epoch: " + str(nofepoch))
subp.plot(xrange,[ val_loss[i] for i in xrange ], '+-',label="val_loss")
subp.plot(xrange,[ loss[i] for i in xrange ], 'o-',label="loss")
subp.plot(xrange,[ 0 for i in range(len(xrange)) ], label="ideal")
subp.legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')            

subp = fig.add_subplot(222)
plt.title("mean algoritmic loss (male) vs. number of data")
subp.plot(xrange,[ float(male_tr[i]) for i in xrange ], '+-',label="male_tr")
subp.plot(xrange,[ float(male_te[i]) for i in xrange ], 'o-',label="male_te")
subp.plot(xrange,[ 0 for i in range(len(xrange)) ], label="ideal")
subp.legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')     

subp = fig.add_subplot(223)
plt.title("Rsq vs. number of data")
subp.plot(xrange,[ float(Rsq_tr[i]) for i in xrange ], '+-',label="Rsq_tr")
subp.plot(xrange,[ float(Rsq_te[i]) for i in xrange ], 'o-',label="Rsq_te")
subp.plot(xrange,[ 1 for i in range(len(xrange)) ], label="ideal")
subp.legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')           

subp = fig.add_subplot(224)
plt.title("maen absolute percent error (mape) (in %) vs. number of data")
subp.plot(xrange,[ float(mape_tr[i]) for i in xrange ], '+-',label="mape_tr")
subp.plot(xrange,[ float(mape_te[i]) for i in xrange ], 'o-',label="mape_te")
subp.plot(xrange,[ 0 for i in range(len(xrange)) ], label="ideal")
subp.legend(bbox_to_anchor=(1.0, 0.0),loc='lower left')  

plt.tight_layout()          

pngfn=os.path.join(rdir,'learning_curves.png')
plt.savefig(pngfn)

plt.show()

                                                                       
for i in dirs:
#        if os.path.isdir(os.path.join('record',i)) and isinstance(int(i), int):
        if os.path.isdir(os.path.join(rdir,str(i))):
            hf=os.path.join(rdir,str(i),'history.csv')
            metrics=os.path.join(rdir,str(i),'metrics.dat')
            if os.path.isfile(hf) and os.path.isfile(metrics):
#                for j in os.listdir(os.path.join(rdir,dirs[int(i)])):
                 for j in os.listdir(os.path.join(rdir,str(i))):
                            #print('*****i,j****',i,j)
                            if len(j.split('.png'))==2:
                                fni=j.split('.png')[0]
                                if fni!='history':
                                    png=fni+'.png'
                                    #png=os.path.join(rdir,dirs[int(i)],png)
                                    png=os.path.join(rdir,str(i),png)
                                    print(png,os.path.isfile(png))
#                        png=os.listdir(os.path.join('record',dirs[int(i)]))[-2]
#                        png=os.path.join('record',i,png)
#                        print(png,os.path.isfile(png))
                        #img=Image(url=png)
                        #img = Image.open(png)
                        #img.show()
                                    display(Image(filename=png))