# Heart-Failure-Prediction-Machine-Learning

More than 300,000 Americans will die this year of sudden cardiac arrest, when the heart abruptly stops working. These events happen suddenly and often without warning, making them nearly impossible to predict. Numerous studies have tried to implement machine learning as a useful tool that can help professionals with predicting heart failure mortality. People with cardiovascular disease or at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia among others) need early detection and management wherein a machine learning model can be of great help. 
This report uses clinical records of patients who had heart failures or complications collected during a follow-up period to see if patients survived or died during such time. 

Machine Learning model that accurately predicts the mortality caused by Heart Failure using medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 12 clinical features.

In this project I first analyze and preprocess the patients’ data to later be used by machine learning models. The code developed in python takes care of cleaning the data to later apply two machine learning techniques: 
-	Artificial Neural Network Classification Model. 
-	Support Vector Machine Classification Model
These models are perfect for binary classification. Given a patient’s clinical records, the models help predict mortality in patients that suffer from heart failure or other heart complications. 
During the learning method phase of this project, models are optimized to gain the best possible results & performance. After improving the models’ parameters and hyperparameters to obtain the best case for each model, the results and metrics obtained are presented to the reader in figures and tables. Such results are analyzed in the discussion section and used to make a final decision between which model makes more sense for the problem that this project tries to solve .


For this task I used the dataset contained in a CSV file from Kaggle called: (heart_failure_clinical_records_dataset.csv) which holds the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 12 clinical features. The original dataset version was collected by Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, and Muhammad Ali Raza (Government College University, Faisalabad, Pakistan) and made available by them on FigShare under the Attribution 4.0 International (CC BY 4.0: freedom to share and adapt the material) copyright in July 2017.
The first thing to do after examining the dataset is to create a function called  cleanse_data that takes a Pandas’ Data Frame as input and returns another Data Frame that was identical to the original except for the invalid rows that are removed using .dropna() to drop the rows that contain blank entries or non-numerical objects in any column. cleanse_data also eliminates rows that contained negative entries for any column by selecting the rows containing only values equal or bigger than 0 and takes that as our new data. This is because you can’t have negative number of platelets in blood or negative age for example. 
My dataset consists of:
•	Features:  
- age: age of the patient [years]
- anaemia: decrease of red blood cells or hemoglobin [Boolean]
- high blood pressure: if the patient has hypertension [Boolean]
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood [mcg/L]
- diabetes: if the patient has diabetes [Boolean]
- ejection fraction: percentage of blood leaving the heart at each contraction [percentage]
- platelets: platelets in the blood [kilo platelets/mL]
- sex: woman or man [binary]
- serum creatinine: level of serum creatinine in the blood [mg/dL]
- serum sodium: level of serum sodium in the blood [mEq/L]
- smoking: if the patient smokes or not [Boolean]
- time: Follow-up period (days)

•	Response: 
- death event: if the patient deceased during the follow-up period [Boolean]
[Boolean values: 0 = Negative (No); 1 = Positive (Yes)]


The code shows two machine learning methods to classify samples as surviving or not surviving heart failure :
-	Artificial Neural Network: Implements MLPClassifier which is a method obtained from scikit-learns’ artificial neural network library. It is based on the multi-layer perceptron algorithm and uses backpropagation. This model utilizes clinical features and corresponding labels to train and predict between 2 classes (surviving or not surviving). For the model’s activation function I chose the logistic function since we are dealing with binary classification.

Due to high varying values in features, a pipeline was created that first scales the data using StandardScaler from scikit-learns’s preprocessing library and then calls the classifier.

-	Support Vector Machine: uses SVC which is a method obtained from scikit-learns’ support vector machine library. It is based on maximizing the classification margins. I chose the kernel type to be ‘rbf’ and  default values for hyperparameters C & gamma. Due to highly varying values in our features,  I created a pipeline that first scales the data using MinMaxScaler from scikit-learns’s preprocessing library and then calls the SVC. 

After creating the pipeline, I used GridSearchCV from scikit-learns’ model selection package, to perform hyperparameter tunning. I gave a range of values for gamma and C to avoid long time searching and then used .best_estimator_ to select the model with the best parameters based on the model’s score.
To validate the models, I created a systematic table that feeds off from a custom function called eval_model. The table displays the name of the model along with its training and test accuracy. With the help of  .score, eval_model evaluates the pipelines of both models and obtains its training & test accuracy. Since accuracy is not enough to compare our binary classification models, I created another table using scikit-learn’s metrics package to look at: precision, recall, f1-score, and ROC AUC score. It is important to know the number of weights of each model to get a better look at the complexity of each method.
All the metrics mentioned in this section along with the complexity of the models will help me decide which model will be best for this project.
