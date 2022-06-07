"""
Created on Mon Mar 21 19:50:58 2022

@author: cnava
"""
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report







#DATA CLEANING / PREPROCESSING FUNCTION
def cleanse_data  (data):
 
 data.dropna()
 data= data[(data >= 0).all(axis=1)]
 

 #print(data)
 #print(data.describe())
 return data

df = cleanse_data(pd.read_csv("heart_failure_clinical_records_dataset.csv"))




#SYSTEMATIC EVALUATION FUNCTION
 
def eval_model (pipe_line, X_train, y_train, X_test, y_test, name):
     d = dict();
     d[1]   = [name,  pipe_line.score(X_test, y_test), pipe_line.score(X_train, y_train)]     
     return d
 


features = ['age', 
            'anaemia', 
            'creatinine_phosphokinase', 'diabetes', 
            'ejection_fraction', 
            'high_blood_pressure', 
            'platelets', 
            'serum_creatinine', 
            'serum_sodium', 
            'sex',  
            'smoking', 'time'] 


response = 'DEATH_EVENT'

X = df[features]
y = df[response]



X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.8, random_state=42 
                                                    )





##ANN
scaler=StandardScaler()
mlp=MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=([100]), batch_size= 25, alpha=0.01, max_iter=500, random_state=0)
pipe=Pipeline([('scaler',StandardScaler()), ('estimator',mlp)])
pipe.fit(X_train,y_train)


print("\n\nANN Accuracy on the training set: {:.2f}".format(pipe.score(X_train, y_train)))
print("ANN Accuracy on the test set: {:.2f}".format(pipe.score(X_test, y_test)))


from sklearn.metrics import ConfusionMatrixDisplay

y_pred = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)
confusion_matrix_ANN= ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print(confusion_matrix_ANN)
#print(pipe.named_steps["estimator"].coefs_)
print("\n\nClassfication Report ANN:")
print(classification_report(y_test, y_pred))
result_ANN = eval_model (pipe, X_train, y_train, X_test, y_test, "ANN")


print('\n\nFITTING REPORT ANN:')
print('Metric\t\t\t\t\t\tTraining\t  Test')
print('\taccuracy            :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.accuracy_score(y_train, y_pred_train), 
       sklearn.metrics.accuracy_score(y_test, y_pred)))
      
print('\tbalanced accuracy   :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.balanced_accuracy_score(y_train, y_pred_train),
       sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

print('\tprecision           :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.precision_score(y_train, y_pred_train),
       sklearn.metrics.precision_score(y_test, y_pred)))

print('\trecall              :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.recall_score(y_train, y_pred_train),
       sklearn.metrics.recall_score(y_test, y_pred)))

print('\tF1 score            :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.f1_score(y_train, y_pred_train),
       sklearn.metrics.f1_score(y_test, y_pred)))

print('\tROC AUC score       :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.roc_auc_score(y_train, y_pred_train),
       sklearn.metrics.roc_auc_score(y_test, y_pred)))






































##SVC
# create pipeline
estimators = [('scaler', MinMaxScaler()),
              ('svm', SVC(kernel='rbf'))]
pipe = Pipeline(estimators)

# define and execute grid search
params = {'svm__C' : 10.0 ** np.arange(-2,3), 'svm__gamma' : np.logspace(-3,1, num=5)}
clfs = GridSearchCV( Pipeline(estimators), params, cv=8, scoring='accuracy') 
clfs.fit(X_train, y_train)

print('\n\n\n\n\n\n\n\n\nSVC Best parameters: ', clfs.best_params_)
print('SVC Best parameter C: ', clfs.best_params_ ['svm__C'])
print('SVC Best parameter gamma: ', clfs.best_params_ ['svm__gamma'])
clfs=clfs.best_estimator_
print("SVC Accuracy on the training set: {:.2f}".format(clfs.score(X_train, y_train)))
print("SVC Accuracy on the test set: {:.2f}".format(clfs.score(X_test, y_test)))
y_pred = clfs.predict(X_test)
y_pred_train = clfs.predict(X_train)
confusion_matrix_svc= ConfusionMatrixDisplay.from_estimator(clfs, X_test, y_test)
print(confusion_matrix_svc)
#print(clfs.named_steps["svm"].dual_coef_)
print("\n\nClassfication Report SVC:")
print(classification_report(y_test, y_pred))
result_SVC = eval_model (clfs, X_train, y_train, X_test, y_test, name="SVC")



print('\n\nFITTING REPORT SVC')
print('Metric\t\t\t\t\t\tTraining\t  Test')
print('\taccuracy            :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.accuracy_score(y_train, y_pred_train), 
       sklearn.metrics.accuracy_score(y_test, y_pred)))
      
print('\tbalanced accuracy   :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.balanced_accuracy_score(y_train, y_pred_train),
       sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

print('\tprecision           :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.precision_score(y_train, y_pred_train),
       sklearn.metrics.precision_score(y_test, y_pred)))

print('\trecall              :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.recall_score(y_train, y_pred_train),
       sklearn.metrics.recall_score(y_test, y_pred)))

print('\tF1 score            :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.f1_score(y_train, y_pred_train),
       sklearn.metrics.f1_score(y_test, y_pred)))

print('\tROC AUC score       :    %6.4f\t\t %6.4f' % 
      (sklearn.metrics.roc_auc_score(y_train, y_pred_train),
       sklearn.metrics.roc_auc_score(y_test, y_pred)))










# PRINTED TABLE ACCURACY.
print("\n\n\n\n\n\n\n\n\nACCURACY TABLE")
results = dict();
results[1]   = result_ANN[1]
results[2]   = result_SVC[1]
print ("{:<20} {:<20} {:<20}".format('name', 'Accuracy_test','Accuracy_train'))
# print each data item.
for key, value in results.items():
    name, Accuracy_test, Accuracy_train = value
    print ("{:<20} {:<20} {:<20}".format(name, Accuracy_test, Accuracy_train))





#BAR CHARTS & HISTOGRAMS
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
variables = ['age', 
            'anaemia', 
            'creatinine_phosphokinase', 'diabetes', 
            'ejection_fraction', 
            'high_blood_pressure', 
            'platelets', 
            'serum_creatinine', 
            'serum_sodium', 
            'sex', 
            'smoking', 'time', 'DEATH_EVENT' ]
vals = [95, 1, 7861, 1, 80, 1, 850000, 9.4, 148, 1, 1, 285, 1]
ax.bar(variables,vals, width=0.9)
plt.title('Max value of each variable')
plt.xlabel('Variables')
plt.ylabel('Max Value')
plt.xticks(rotation=30, ha='right')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
variables = ['age', 
            'anaemia', 
            'creatinine_phosphokinase', 'diabetes', 
            'ejection_fraction', 
            'high_blood_pressure', 
            'platelets', 
            'serum_creatinine', 
            'serum_sodium', 
            'sex', 
            'smoking', 'time', 'DEATH_EVENT' ]
vals = [40, 0, 23, 0, 14, 0, 25100, 0.5, 113, 0, 0, 4, 0]
ax.bar(variables,vals, width=0.9)
plt.title('Min value of each variable')
plt.xlabel('Variables')
plt.ylabel('Min Value')
plt.xticks(rotation=30, ha='right')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
variables = ['age', 
            'anaemia', 
            'creatinine_phosphokinase', 'diabetes', 
            'ejection_fraction', 
            'high_blood_pressure', 
            'platelets', 
            'serum_creatinine', 
            'serum_sodium', 
            'sex', 
            'smoking', 'time', 'DEATH_EVENT' ]
vals = [60.833893, 0.431438, 581.839465, 0.41806, 38.083612, 0.351171, 263358.0293, 1.39388, 136.625418, 0.648829, 0.32107, 130.26, 0.32107]
ax.bar(variables,vals, width=0.9)
plt.title('Mean value of each variable')
plt.xlabel('Variables')
plt.ylabel('Mean Value')
plt.xticks(rotation=30, ha='right')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
variables = ['age', 
            'anaemia', 
            'creatinine_phosphokinase', 'diabetes', 
            'ejection_fraction', 
            'high_blood_pressure',  
            'serum_creatinine', 
            'serum_sodium', 
            'sex', 
            'smoking', 'time', 'DEATH_EVENT' ]
vals = [60.833893, 0.431438, 581.839465, 0.41806, 38.083612, 0.351171, 1.39388, 136.625418, 0.648829, 0.32107, 130.26, 0.32107]
ax.bar(variables,vals, width=0.9)
plt.title('Mean value of each variable')
plt.xlabel('Variables')
plt.ylabel('Mean Value')
plt.xticks(rotation=30, ha='right')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
variables = ['Survived', 'Not Survived' ]
vals = [203, 96]
ax.bar(variables,vals, width=0.5)
plt.title('Label Count')
plt.xlabel('Death_Event')
plt.ylabel('count')
plt.xticks(rotation=30, ha='right')
plt.show()