import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import logging as logger

logger.basicConfig(filename='tf_classifier_demo.log', level=logger.INFO)
#
# import data
housing_df = pd.read_csv('census_data.csv')
# print(housing_df.dtypes)
# print(housing_df.head())
y_val = housing_df['income_bracket'].apply(lambda x: 1 if (str(x).strip() in '>50K') else 0)
x_data = housing_df.drop('income_bracket', axis=1)
cols = x_data.columns
# print(cols)
#
# Split data 70/30
X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.30, random_state=101)
#
#
# FEATURE COLUMNS
# Numeric
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
education_num = tf.feature_column.numeric_column('education_num')
age = tf.feature_column.numeric_column('age')
#
# Categorical
gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['Female','Male'])
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)
race = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=1000)
#
# Bucketize 'Age' Column in DF  (continuous to bucketise
# age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])
#
feat_cols = [capital_gain, capital_loss, hours_per_week, age,education_num,
             workclass, education, marital_status, occupation, relationship,race,gender,native_country]
#
# CREATE THE MODEL USING LinearClassifier (Binary Classifier)
# https://www.guru99.com/linear-classifier-tensorflow.html
'''
Classification problems represent roughly 80 percent of the machine learning task. 
Classification aims at predicting the probability of each class given a set of inputs. 
The label (i.e., the dependent variable) is a discrete value, called a class.

 -- If the label has only two classes, the learning algorithm is a binary classifier.
 -- Multiclass classifier tackles labels with more than two classes.
For instance, a typical binary classification problem is to predict the likelihood a customer makes a second purchase. 
Predict the type of animal displayed on a picture is multiclass classification problem since there are more than two varieties of animal existing
'''
print(f'Begin Create LinearClassifier model x=X_train, y=y_train:   {datetime.datetime.now()}')
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=128, num_epochs=100, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)  # the label (predicted value is 2 levels)
model.train(input_fn=input_func, steps=10000)
print(f'End Create LinearClassifier model x=X_train, y=y_train:   {datetime.datetime.now()}')
#
# EVALUATE THE MODEL by re-using Train Data
print(f'Start LinearClassifier predicting using x=X_train:   {datetime.datetime.now()}')
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=128, num_epochs=100, shuffle=False)
predictions_train = model.evaluate(input_fn=eval_input_func, steps=1000)
#
# PREDICT USING THE MODEL Test Data
print(f'Start Predict LinearClassifier using x=X_test:   {datetime.datetime.now()}')
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=128, num_epochs=100, shuffle=False)
predictions = model.predict(input_fn = pred_input_func)
preds = list(predictions)
# print(preds)
print(f'End Predict LinearClassifier using x=X_test:   {datetime.datetime.now()}')
y_pred = [pred['class_ids'][0] for pred in preds]
target_names = ['y_test', 'y_pred']
print(classification_report(y_test, y_pred[-len(X_test):], target_names=target_names))
##
## USE DNNLinearClassifier instead of LinearClassifier
#
# Create the feature column
embedded_gender = tf.feature_column.embedding_column(gender, dimension=len(housing_df['gender'].unique()))
embedded_occupation = tf.feature_column.embedding_column(occupation, dimension=len(housing_df['occupation'].unique()))
embedded_marital_status = tf.feature_column.embedding_column(marital_status, dimension=len(housing_df['marital_status'].unique()))
embedded_relationship = tf.feature_column.embedding_column(relationship, dimension=len(housing_df['relationship'].unique()))
embedded_education = tf.feature_column.embedding_column(education, dimension=len(housing_df['education'].unique()))
embedded_workclass = tf.feature_column.embedding_column(workclass, dimension=len(housing_df['workclass'].unique()))
embedded_native_country = tf.feature_column.embedding_column(native_country, dimension=len(housing_df['native_country'].unique()))
embedded_race = tf.feature_column.embedding_column(race, dimension=len(housing_df['race'].unique()))
feat_cols = [capital_gain, capital_loss, hours_per_week, age,education_num,
             embedded_workclass, embedded_education, embedded_marital_status, embedded_occupation,
             embedded_relationship,embedded_race,embedded_gender,embedded_native_country]
#
# CREATE THE MODEL
print(f'Begin Create DNNClassifier model usng x=X_train, y=y_train:   {datetime.datetime.now()}')
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=128, num_epochs=100, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(feature_columns=feat_cols,hidden_units=[13,13,13, 13], n_classes=2)  # the label (predicted value is 2 levels)
dnn_model.train(input_fn=input_func, steps=1000)
print(f'End Create DNNClassifier model using x=X_train, y=y_train:   {datetime.datetime.now()}')
#
# EVALUATE THE MODEL by re-using Train Data
print(f'Begin predict DNNClassifier using x=X_train:   {datetime.datetime.now()}')
eval_input_func_dnn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=128, num_epochs=100, shuffle=False)
predictions_train_dnn = dnn_model.evaluate(input_fn=eval_input_func_dnn,steps=1000)
#
# PREDICT USING THE MODEL
print(f'Begin Predict DNNClassifier using x=X_test:   {datetime.datetime.now()}')
pred_input_func_dnn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=128, num_epochs=100, shuffle=False)
predictions_dnn = dnn_model.predict(pred_input_func_dnn)
preds_dnn = list(predictions_dnn)
# print(preds)
print(f'End Predict DNNClassifier x=X_train, y=y_train:   {datetime.datetime.now()}')
y_pred_dnn = []
for i in range(len(y_test)):
    y_pred_dnn.append(preds_dnn[i]['class_ids'][0])
target_names = ['y_test', 'y_pred']
print(classification_report(y_test, y_pred_dnn[-len(X_test):], target_names=target_names))