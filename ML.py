#Machine Learning for Breast Cancer Detecttion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#There are 30 numeric, predictive attributes in the dataset of Breast Cancer Detection
load = pd.read_csv(r"C:\Users\Gabriela Navas\Documents\Python\pythonwithgabs\data.csv")
load.head()
load.shape  #to see dimension of the dataset
load.columns #to display the column names in the dataset
load.info()  #display information about the column attributes
load.isna().sum()  #counting the empty values in each column
load= load.drop('Unnamed: 32', axis=1) #since we have no values on the unnamed column, we must drop it off
load["diagnosis"].value_counts()# To count the values on the diagnosis column
x=list(load.columns) #Converting the column names to a list
print(x)
load.describe() #a summary of all numeric columns
#A visualization that will show the total count of malignant and benign patients in a counterplot
sns.countplot(load["diagnosis"]);
#heatmap of Correlation 
corr=load.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr);
load['diagnosis']= load['diagnosis'].map({"M":1,"B":0})
load["diagnosis"].unique()

#Malignant type (1)
#Benign type(0)