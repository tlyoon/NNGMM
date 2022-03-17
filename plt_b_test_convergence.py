import os
import pandas as pd
import csv, numpy as np
from collections import Counter
from matplotlib import pyplot as plt

fn='b_test_convergence.csv'
f=open(fn,'r+')
data=f.readline()
header=data.split(" ")[0:5]

data=[ i for i in data.split(" ") if i !='']
data=data[5:]
data=[ i.split(",") for i in data ]

allindex=[ j[0] for j in data ]

diccount=Counter(allindex)

indexdir=np.sort(list(set([ i[0] for i in data ])))
accum={}
for j in indexdir:
    accum[j]=0

for i in data:
    for j in indexdir:
        if i[0]==j:
            accum[j]=accum[j]+float(i[-1])
    #print('')

xconverg=[];yconverg=[]    
indexdir=np.sort([ int(i) for i in indexdir ])
for j in indexdir:
    avergage=accum[str(j)]/diccount[str(j)]
    print(j,diccount[str(j)],accum[str(j)],avergage)
    xconverg.append(int(j))
    yconverg.append(avergage)

plt.plot(xconverg,yconverg,'.-')
plt.title('accuracy vs. b_models')
pngfn=os.path.join('b_test_convergence.png')
plt.savefig(pngfn)
