classes = [0,1,2,3,4,5,6,7,8,9]
confusion_matrix = {c:{c1:0 for c1 in classes} for c in classes}
f1 = open("predicted.txt", 'r')
f2 = open("correct_labels.txt", 'r')
i=0
f=open("misclassified.txt",'w')
for predicted_class,actual_class in zip(f1,f2):
    
    confusion_matrix[int(actual_class.strip())][int(predicted_class.strip())] += 1
    if(int(actual_class.strip())==9 and int(predicted_class.strip())!=9):
        f.write(str(i) + " " + predicted_class.strip()+"\n")

    i+=1
f.close()
for i in classes:
    for j in classes:
        print(confusion_matrix[i][j], end=" ")
    print("")