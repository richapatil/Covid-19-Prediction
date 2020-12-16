#importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score as rs
from sklearn.metrics import precision_score as ps
from sklearn.metrics import f1_score as fs

encoder = preprocessing.LabelEncoder()


#importing dataset
data = pd.read_csv('final (1).csv')
data = data.fillna(np.nan,axis=0)

#encoding catagorical values

data['location'] = encoder.fit_transform(data['location'].astype(str))
data['country'] = encoder.fit_transform(data['country'].astype(str))
data['gender'] = encoder.fit_transform(data['gender'].astype(str))
data[['fever']] = encoder.fit_transform(data['fever'].astype(str))
data[['dry cough']] = encoder.fit_transform(data['dry cough'].astype(str))
data[['difficulty in breathing']] = encoder.fit_transform(data['difficulty in breathing'].astype(str))
data[['tiredness']] = encoder.fit_transform(data['tiredness'].astype(str))
data[['soar_throat']] = encoder.fit_transform(data['soar_throat'].astype(str))
data[['nasal_congestion']] = encoder.fit_transform(data['nasal_congestion'].astype(str))

#printing datatype of each column
print(data.dtypes)


#printing first few entry
print(data.head())

#finding out null values
print(data.isna().sum())


#importing libraries
from sklearn.metrics import recall_score as rs
from sklearn.metrics import precision_score as ps
from sklearn.metrics import f1_score as fs
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.metrics import confusion_matrix as cm



#Extracting the data set into X and Y values
X = data[['gender','age','fever','dry cough','difficulty in breathing','tiredness','soar_throat','nasal_congestion','diff_symptoms']]
Y = data['result']


#Spliting data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# TRaining the model
rf= RandomForestClassifier(n_estimators=50, random_state=1)
rf.fit(X,Y)
#rf.fit(X_train,np.array(Y_train).reshape(Y_train.shape[0],1))

#predicting the values
pred = np.array(rf.predict(X_test))

recall = rs(Y_test,pred)
precision = ps(Y_test,pred)
f1 = fs(Y_test,pred)
ma = rf.score(X_test,Y_test)

#Printing All score
print('*** Evaluation metrics for test dataset ***\n')
print('Recall Score: ',recall)
print('Precision Score: ',precision)
print('F1 Score: ',f1)
print('Accuracy: ',ma)
a = pd.DataFrame(Y_test)
a['pred']= rf.predict(X_test)
print('\n\tTable 3\n')
print(a.head())


# we can print the accuracy of the classification problem and create the confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(Y_test,pred)
print(cm)



# Now we will use the mlxtend library to plot the confusion matrix
# and to visualize the output
#from mlxtend.plotting import plot_confusion_matrix,plot_decision_regions
#plot_confusion_matrix(cm)
#plt.show()




#Visulization of output varible

#plt.bar(['Accuracy','F1 Score','Recall Score','Precision Score'],[ma,f1,recall,precision],color=['red','green','purple','orange'])
#plt.plot([ma,f1,recall,precision],color='black')
#plt.title('Evaluation Metrics for Boosted Random Forest ')



# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
import pickle
pickle.dump(rf, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,45,1,1,1,1,1,0,1]]))
