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
        
        if predict_only:
            self.loadTrainingData(model_name)
            self.out_filename = test_file_out
            self.test_x = self.ReadTextFile(test_file)
        else:
            self.train_x,self.train_y = self.ReadTextFile(train_file)
            self.test_x,self.test_y = self.ReadTextFile(test_file)
            
            self.m = len(self.train_x)                     #number of examples
            self.n = len(self.train_x[0])                  #number of features
            self.C = C
            self.classes = sorted(list(set(self.train_y)))
            self.num_classes = len(self.classes)   #number of classes
            self.threshold = 0.0001
    
    def storeTrainingData(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.classes, f)
        pickle.dump(self.params, f)
        f.close()
    
    def loadTrainingData(self, filename):
        f = open(filename, 'rb')
        self.classes = pickle.load(f)
        self.params = pickle.load(f)
        self.num_classes = len(self.classes)   #number of classes
        f.close()
    
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
    
    def Pegasos(self, X, Y, k, T):
        w_t = np.zeros(len(X[0]))
        w_t1 = w_t.copy()
        b = 0
        m  = len(X)
        #print(X)
        for t in range(1,T+1):
            w_t1 = w_t.copy()
            A_t = random.sample(range(m),k)
            yi_xi = 0
            yi = 0
            for i in A_t:
                if(Y[i]*(w_t.T.dot(X[i])+b)<1.0):
                    yi_xi += Y[i]*X[i]
                    yi += Y[i]
            eta_t = 1/t
            #update step
            delL = w_t-self.C*yi_xi
            w_t = w_t - eta_t*delL
            b = b + self.C*eta_t*yi
            
            #if(np.max(w_t-w_t1)<self.threshold):
            #   break
        #print(np.max(w_t-w_t1),t)
        return (w_t,b)
    
    def SplitClassSamples(self):
        self.samples = {c: self.train_x[np.where(self.train_y==c)] for c in self.classes}
        
    def MultiClassPegasos(self):
        self.Ys_c1 = {c: np.ones(self.samples[c].shape[0]) for c in self.classes}
        self.Ys_c2 = {c: np.full(self.samples[c].shape[0],-1, dtype=int) for c in self.classes}
        
        self.params = {c:{} for c in self.classes}
        
        for i in range(self.num_classes):
            for j in range(i+1,self.num_classes):
                c1 = self.classes[i]
                c2 = self.classes[j]
                
                X = np.concatenate((self.samples[c1],self.samples[c2]))
                Y = np.concatenate((self.Ys_c1[c1],self.Ys_c2[c2]))
                self.params[c1][c2] = self.Pegasos(X, Y, k=100, T=1000)
    
    def MultiClassTesting(self, data_type='test'):
        if(self.predict_only):
            data_x = self.test_x
        elif(data_type=='test'):
            data_x = self.test_x
            data_y = self.test_y
        elif(data_type=='train'):
            data_x = self.train_x
            data_y = self.train_y
        
        correct_count = 0            #variable to count correct predictions
        
        counts_bck = {c:0 for c in self.classes}
        if(self.predict_only):
            f = open(self.out_filename, 'w')
            for ex in data_x:
                counts = copy(counts_bck)
                for i1 in range(self.num_classes):
                    for i2 in range(i1+1,self.num_classes):
                        c1 = self.classes[i1]
                        c2 = self.classes[i2]
                        
                        if(self.params[c1][c2][0].dot(ex) + self.params[c1][c2][1] >= 0):
                            counts[c1]+=1
                        else:
                            counts[c2]+=1
                
                max_score = max(counts.items(), key=lambda a:a[1])[1]
                predicted_class = max(filter(lambda a:a[1]==max_score,counts.items()), key=lambda a:a[0])[0]
                f.write(predicted_class)
                f.write('\n')
            f.close()
        else:
            for actual_class,ex in zip(data_y,data_x):
                counts = copy(counts_bck)
                for i1 in range(self.num_classes):
                    for i2 in range(i1+1,self.num_classes):
                        c1 = self.classes[i1]
                        c2 = self.classes[i2]
                        
                        if(self.params[c1][c2][0].dot(ex) + self.params[c1][c2][1] >= 0):
                            counts[c1]+=1
                        else:
                            counts[c2]+=1
                
                max_score = max(counts.items(), key=lambda a:a[1])[1]
                predicted_class = max(filter(lambda a:a[1]==max_score,counts.items()), key=lambda a:a[0])[0]
                if(predicted_class == actual_class):
                    correct_count += 1
            
            accuracy = 1.0*correct_count/len(data_x)
            return accuracy
        
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
    
    if predict_only:
        svm.MultiClassTesting()
        print("Prediction complete...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    else:
        svm.SplitClassSamples()      #split the training data according to classes
        svm.MultiClassPegasos()
        print("Pegasos training complete...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        svm.storeTrainingData(svm.model_name)
        print("Saved training data...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        accuracy = svm.MultiClassTesting('train')
        print("Train data accuracy: %.2f%%"%(accuracy*100))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        accuracy = svm.MultiClassTesting('test')
        print("Test data accuracy: %.2f%%"%(accuracy*100))
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
        
        print("Pegasos testing complete...")
    
    
    
    