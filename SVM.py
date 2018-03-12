import numpy as np
from copy import copy
import pickle
import random
import sys

from timeit import default_timer as timer

class MultiClassSVM:
    
    def __init__(self, train_file="", test_file="", test_file_out="", C=1, model_name="", predict_only=False):
        self.predict_only = predict_only
        self.model_name = model_name
        
        self.train_x,self.train_y = self.ReadTextFile(train_file)
        self.test_x,self.test_y = self.ReadTextFile(test_file)
        
        self.m = len(self.train_x)                     #number of examples
        self.n = len(self.train_x[0])                  #number of features
        self.C = C
        self.classes = sorted(list(set(self.train_y)))
        self.num_classes = len(self.classes)   #number of classes
        self.threshold = 0.0001
    
    
    #function to read file having input data
    def ReadTextFile(self, file_name):
        fin = open(file_name, 'r')
        data_x = []
        data_y = []
        if(self.predict_only):
            for inp in fin:
                data = inp.rstrip().split(',')
                data_x.append(data)
        else:
            for inp in fin:
                data = inp.rstrip().split(',')
                data_x.append(data[:-1])
                data_y.append(data[-1])
        fin.close()
        
        if(self.predict_only):
            return np.array(data_x,dtype=float)/255
        else:
            return (np.array(data_x,dtype=float)/255,np.array(data_y))

        
def calculate_time_elapsed():
    global start
    time_elapsed = timer()-start
    start = timer()
    return time_elapsed
    
if __name__=='__main__':
    
    start = timer()
    
    #create a SVM object
    predict_only = False
    run_libsvm = False
    if(len(sys.argv)==1):
        svm = MultiClassSVM("train.csv","test.csv",1, model_name="model0")
    elif(len(sys.argv)==5 and sys.argv[1]=="-p"):
        predict_only = True
        svm = MultiClassSVM(test_file=sys.argv[2],test_file_out=sys.argv[3], model_name=sys.argv[4], predict_only=True)
    elif(len(sys.argv)==4):
        svm = MultiClassSVM(sys.argv[1],sys.argv[2], model_name=sys.argv[3])
    else:
        print("Invalid arguments.")
        print("Usage: SVM.py [-p] [training_file_with_labels] testing_file_inp [testing_file_out] model_name")
        print("options:")
        print("-p: Prediction only.")
        sys.exit(1)
        
    print("Reading data complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    
    
    