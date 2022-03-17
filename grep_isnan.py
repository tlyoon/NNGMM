import numpy as np
import os

dirs=[ i for i in os.listdir() if len(i.split('config'))==2 ]
for i in dirs:
    print(i)
    data=np.load(i)
#    dd=data.flatten('F')
#    out=[ i for i in dd if np.isnan(i) ]

    array_sum = np.sum(data)
    array_has_nan = np.isnan(array_sum)
    print(array_has_nan)

#    print(out)
