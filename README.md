# Heart-Failure-Prediction-Machine-Learning
Machine Learning model that accurately predicts the mortality caused by Heart Failure using medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 12 clinical features.

In this project I first analyze and preprocess the patients’ data to later be used by machine learning models. The code developed in python takes care of cleaning the data to later apply two machine learning techniques: 
-	Artificial Neural Network Classification Model. 
-	Support Vector Machine Classification Model
These models are perfect for binary classification. Given a patient’s clinical records, the models help predict mortality in patients that suffer from heart failure or other heart complications. 
During the learning method phase of this project, models are optimized to gain the best possible results & performance. After improving the models’ parameters and hyperparameters to obtain the best case for each model, the results and metrics obtained are presented to the reader in figures and tables. Such results are analyzed in the discussion section and used to make a final decision between which model makes more sense for the problem that this project tries to solve .


I am tasked is to create a model that accurately predicts the mortality caused by Heart Failure. For such task I used the dataset contained in a CSV file from Kaggle called: (heart_failure_clinical_records_dataset.csv) which holds the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 12 clinical features. The original dataset version was collected by Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, and Muhammad Ali Raza (Government College University, Faisalabad, Pakistan) and made available by them on FigShare under the Attribution 4.0 International (CC BY 4.0: freedom to share and adapt the material) copyright in July 2017.
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


