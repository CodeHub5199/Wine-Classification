import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# model = KNeighborsClassifier(n_neighbors=1) # As n_neighbors increases, accuracy score decreases
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print('Accuracy score: ',score)

'''
KNeighborsClassifier
Accuracy score : 0.7777777777777778
    
DecisionTreeClassifier
Accuracy score : 0.9444444444444444
    
RandomForestClassifier
Accuracy score : 1.0
'''

model_list = ['KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier']
accuracy_score_list = [0.7777777777777778,0.9444444444444444,1.0]
plt.figure(figsize=(8,6))
plt.title('Model and Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.bar(model_list,accuracy_score_list,color=['red','blue','green'])
plt.show()

'''
Feature names: 
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 
'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
Target: 
['class_0', 'class_1', 'class_2']
'''
