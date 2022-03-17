import os
import pandas as pd
import csv, numpy as np
from collections import Counter

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

fn='test_convergence.csv'
#f=open(fn,'r+')
#data=f.readlines()
#header=data.split(" ")[0:5]
data=pd.read_csv(fn,sep=' ')

indexdir=np.sort(list(set([ data.iloc[i][0] for i in range(len(data)) ])))

accum={}
for j in indexdir:
    accum[j]=0



numofdata=0;nofdataarr=[]
amapearr=[];amalearr=[];
arsqarr=[];amsearr=[]
nnaddic={};

for j in list(indexdir):
    metricpath=os.path.join(str(j),'metrics.dat')
    historypath=os.path.join(str(j),'history.png')
    if os.path.isfile(metricpath) and os.path.isfile(historypath):
        count=0
        smape=0;smale=0;smse=0;srsq=0
        
        ####
        #print('To abstract number of accumulated training data in model j ',j)     
        f = open(metricpath, "r")
        line=f.readlines()
        nnad=int(line[0].strip().split()[-1])
        nnaddic[j]=nnad
        numofdata=nnad+numofdata
        f.close()
        print('number of non-accumulative training data for model in directory',j,nnad)
        print('number of accumulative training data for model in directory',j,numofdata)
        #print(j,metricpath,)
        ####
        
        for i in range(len(data)):
            if data.iloc[i][0]==j:
                count=count+1
                #numofdata=data.iloc[i][3]+numofdata
                #numofdata=nnad+numofdata
                runno=data.iloc[i][0]
                mape=data.iloc[i][4]
                male=data.iloc[i][5]
                mse=data.iloc[i][6]
                rsq=data.iloc[i][7]            
                #print('i,j',i,j)
                print(runno,mape,male,mse,rsq)
                smape=smape+mape            
                smale=smale+male            
                smse=smse+mse
                srsq=srsq+rsq
                #accum[j]=accum[j]+float(i[-1])
            if count !=0:
                amape=smape/count
                amale=smale/count
                arsq=srsq/count
                amse=smse/count
        print(j,amape,smale,smse,srsq)
    #noftraindataarr.append(nnad)
        nofdataarr.append(numofdata)
        amapearr.append(round(amape,2))
        amalearr.append(round(amale,2))
        arsqarr.append(round(arsq,3))
        amsearr.append(round(amse,2))
        print('')


fig , ax = plt.subplots(nrows = 2, ncols = 2, figsize=(9,6))

ax[0,0].set_title('mape')
ax[0,0].plot(nofdataarr,amapearr,'.-')
ax[0,0].axhline(0,label='ideal',color='r',linestyle='--')
ax[0,0].legend()

ax[0,1].set_title('male')
ax[0,1].plot(nofdataarr,amalearr,'.-')
ax[0,1].axhline(0,label='ideal',color='r',linestyle='--')
ax[0,1].legend()

#ax[1,0]=fig.add_subplot(2,2,3)
ax[1,0].set_title('rsq')
ax[1,0].plot(nofdataarr,arsqarr,'.-')
ax[1,0].axhline(1,label='ideal',color='r',linestyle='--')
ax[1,0].legend()
ax[1,0].set_ylim([0, 1.1])
#ax[1,0].set_ylim(-0.5,1.5)

#ax[1,1]=fig.add_subplot(2,2,4)
ax[1,1].set_title('mse')
ax[1,1].plot(nofdataarr,amsearr,'.-')
ax[1,1].axhline(0,label='ideal',color='r',linestyle='--')
ax[1,1].legend()
plt.tight_layout()

pngfn=os.path.join('test_convergence.png')
plt.savefig(pngfn)

f=open("convergence_performance.csv",'w')
state='"index"  "accumulative number of training data"  "average mse"  "average Rsq"'; 
f.write(state+'\n')
for i in range(len(nofdataarr)):
    state=str(i) + ' ' + str(nofdataarr[i]) + ' ' + str(amsearr[i]) + ' ' + str(arsqarr[i])
    f.write(state + '\n')
f.close()

