## https://stackoverflow.com/questions/788411/check-to-see-if-python-script-is-running

## Mandatory requirement 1: This script works in conjuction with itrain.py. The presence of itrain.py 
## in the same directorory as this script is mandatory

## Mandatory requirement 2: Existence of data_train/data_repo/ containing at least 
## one pair of training data {label.XXX_c_YYY.npy, config.XXX_c_YYY.npy}. 

## Mandatory requirement 3: Existence of data_test/ containing {label.XXX_c_YYY.npy, config.XXX_c_YYY.npy} pairs which are not seen before by the trained model. 
## These data sets must not be the same as that used for training the model.

# 1. This script checks if any {config.*_c_*.npy, label.*_c_*.npy} 
# pair is present in data_train/. If positive, they will be moved to data_train/data_used/.

# 2. This script will then scan data_train/data_repo/ for the 
# first-encountered {config.*_c_*.npy, label.*_c_*.npy} pair in the que 
# and then move them into data_train/

# 3. This script then executes itrain.py using the {config.*_c_*.npy, label.*_c_*.npy} 
## pair in data_train/, which was obtained from step 2 above. The completion of 
## executing itrain.py will take some times, of the time scale ~10^0 hours. 

# 4. The status of the completion in step 3 will be automatically checked at an 60 seconds interval. 
## If execution of itrain.py has not yet completed, no action will be taken. 

# 5. If the execution of itrain.py has completed, the process from step 1 - step 4 will be automatically executed.

# 6. The processes 1 - 5 will continue to iterate automatically as long as there are legitimate pairs 
# {config.*_c.npy_*, label.*_c_*.npy} found in data_train/data_repo/.

# 7. If the legitimate pairs {config.*_c_*.npy, label.*_c_*.npy} in data_train/data_repo/ dry up, the code shall
# be terminated.

import numpy as np
import os,sys
import shutil
import atexit
import time

def executeSomething():
    #code here
    time.sleep(60)

while True:
    executeSomething()
    
    try:
        # Set PID file
        def set_pid_file():
            pid = str(os.getpid())
            f = open('myCode.pid', 'w')
            f.write(pid)
            f.close()
    
        def goodby():
            pid = str('myCode.pid')
            os.remove(pid)
    
        atexit.register(goodby)
        set_pid_file()
    
        # Place your code here
        os.system('./link_npy_fr_repo.sh')
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
                        if os.path.isfile(os.path.join('data_train','data_used',Xtrlist[count])) \
			            or os.path.isfile(os.path.join('data_train','data_used',Ytrlist[count])): 
                            try:
                                os.remove(os.path.join('data_train',Xtrlist[count]))
                            except:
                                zero=0
                            try:
                                os.remove(os.path.join('data_train',Ytrlist[count]))
                            except:
                                zero=0
                            print(Xtrlist[count],Ytrlist[count],'exited in',os.path.join('data_train','data_used'),\
                                  '. They have now been removed from data_train')
                        else:
                            shutil.move(os.path.join('data_train',Xtrlist[count]), \
                                        os.path.join('data_train','data_used'))
                            shutil.move(os.path.join('data_train',Ytrlist[count]),os.path.join('data_train','data_used'))
                            print(Xtrlist[count],Ytrlist[count],'have been moved from data_train into data_train/data_used')
                        count=count+1
        
        for fnconfig in os.listdir(os.path.join('data_train','data_repo')):
            if fnconfig.split('.')[0]=='config':
                nofconfig=fnconfig.split('config.')[1].split('_c')[0]
                bhao=fnconfig.split('config.')[1].split('.npy')[0]
                fnlabel='label.'+ bhao +'.npy'
                fl=os.path.join('data_train','data_repo',fnlabel)
                fc=os.path.join('data_train','data_repo',fnconfig)
                if os.path.isfile(fl) and os.path.isfile(fc):
                    print('check dimensional consistency')
                    print(fl,fc)
                    flload=np.load(fl);fcload=np.load(fc)
                    if fcload.shape[0]==flload.shape[0]:
                        print('dimensions are consistent. Number of rows in both are:',flload.shape[0])
                        print('to move both files to data_train')
                        shutil.move(fl,'data_train')
                        shutil.move(fc,'data_train')
                        print(fl,fc,'moved to data_train/')
                        break
                    print('')
        try:
            if os.path.isfile(os.path.join('data_train',fnlabel)) \
                and os.path.isfile(os.path.join('data_train',fnconfig)):
                    #import train
                    os.system("python itrain.py")
                    #from subprocess import call
                    #call(["python", "train.py"])
            else:
                sys.exit()
                #zero=0
        except:
            print('no data files are found in data_train/. To exit')
            sys.exit()
            #zero=0
        # end Place your code here
    
    except KeyboardInterrupt:
        sys.exit(0)
