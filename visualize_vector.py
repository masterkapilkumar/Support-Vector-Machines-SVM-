import numpy as np
import matplotlib.pyplot as plt

f = open("misclassified.txt", 'r')
f1 = open("predict.csv", 'r')
exs = [list(map(int,line.strip().split(","))) for line in f1]

num=30

for line in f:
    x=list(map(int,line.strip().split()))
    digit = np.array(exs[x[0]])
    digit.resize((28,28))
    plt.imshow(digit, 'gray')
    plt.title(x[1])
    plt.show()
    num-=1
    if(num==0):
        break