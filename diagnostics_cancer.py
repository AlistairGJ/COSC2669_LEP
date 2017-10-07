#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# URL - http://archive.ics.uci.edu/ml/datasets/mammographic+mass

# In[] Importing the data analysis libraries

import pandas as pd
import numpy as np
import urllib2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas.tools.plotting import scatter_matrix
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
set1 = urllib2.Request(url)
breast_p= urllib2.urlopen(set1)
breast = pd.read_csv(breast_p, sep=',', decimal='.', header=None, names=('BIRADS', 'age', 'shape', 'margin', 'density', 'severity'))

# In[] Checking the data
breast.dtypes

# In[] Cleaning the data - BIRADS
mask_unknown = '?'
mask_55 = '55'
breast.loc[breast['BIRADS'] == mask_unknown, 'BIRADS'] = 0
breast.loc[breast['BIRADS'] == mask_55, 'BIRADS'] = 5
breast['BIRADS'] = breast['BIRADS'].str.strip()
breast['BIRADS'].fillna(0, inplace=True)
breast['BIRADS'].value_counts()

# In[] Cleaning the data - Age
breast['age'].value_counts()
mask_unknown = '?'
breast.loc[breast['age'] == mask_unknown, 'age'] = 60
breast['age'] = breast['age'].str.strip()
breast['age'].fillna(60, inplace=True)
breast['age'].value_counts()
## Median breast cancer age of 60 years was used to fill missing

# In[] Cleaning the data - Shape
breast['shape'].value_counts()
mask_unknown = '?'
breast.loc[breast['shape'] == mask_unknown, 'shape'] = 0
breast['shape'] = breast['shape'].str.strip()
breast['shape'].fillna(0, inplace=True)
breast['shape'].value_counts()

# In[] Cleaning the data - Margin
breast['margin'].value_counts()
mask_unknown = '?'
breast.loc[breast['margin'] == mask_unknown, 'margin'] = 0
breast['margin'].value_counts()

# In[] Cleaning the data - Density
breast['density'].value_counts()
mask_unknown = '?'
breast.loc[breast['density'] == mask_unknown, 'density'] = 0
breast['density'].value_counts()

# In[] Cleaning the data - Severity
breast['severity'].value_counts()

# In[] Cleaning the data - Severity
breast.isnull().any()

# In[] Cleaning the data - Data Types Checking
breast.dtypes

# In[] Cleaning the data - Data Types Updating
breast['BIRADS'] = breast['BIRADS'].astype(int)
breast['age'] = breast['age'].astype(int)
breast['shape'] = breast['shape'].astype(int)
breast['margin'] = breast['margin'].astype(int)
breast['density'] = breast['density'].astype(int)

# In[] Cleaning the data - Data Types Checking
breast.dtypes

# In[] Visualising the Data - Bar Graph for BIRADS
breast['BIRADS'].value_counts().plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Ranking')
plt.title('Mass BI-RADS Assessment Scores')
plt.show()

# In[] Exploring the Data - Summary Statistics for Age
age_mean = breast['age'].mean()
age_min = breast['age'].min()
age_max = breast['age'].max()
age_median = breast['age'].median()
age_mode = breast['age'].mode()

# In[] Visualising the Data - Histogram for Age
breast['age'].plot(kind='hist', bins=20, colors=['skyblue'])
plt.title('Distribution of Age Amongst the Sample Population')
plt.xlabel('Age (years)')
plt.show()

# In[] Exploring the Data - Mode for Breast Shape
shape_mode = breast['shape'].mode()
## 4 - this is equivalent to 'irregular'

# In[] Visualising the Data - Pie Chart for Spape
breast['shape'].value_counts().plot(kind='pie', autopct='%.2f', colors=['lightskyblue', 'green', 'red', 'purple', 'yellow'])
blue_patch = mpatches.Patch(color='lightskyblue', label='Irregular')
yellow_patch = mpatches.Patch(color='yellow', label='Unknown Values')
purple_patch = mpatches.Patch(color='purple', label='Lobular')
red_patch = mpatches.Patch(color='red', label='Oval')
green_patch = mpatches.Patch(color='green', label='Round')
plt.title('Mass Shape Proportions Amongst the Sample Population')
plt.xlabel('Mass Shape')
plt.legend(handles=[blue_patch, yellow_patch, purple_patch, red_patch, green_patch])
plt.show()

# In[] Exploring the Data - Mode for Margin
margin_mode = breast['margin'].mode()
## 1 - this is equivalent to margin being 'circumscribed'

# In[] Visualising the Data - Bar Chart for Margin
breast['margin'].value_counts().plot(kind='bar', colors='rgbkym')
r_patch = mpatches.Patch(color='r', label='Circumscribed')
g_patch = mpatches.Patch(color='g', label='Ill-Defined')
b_patch = mpatches.Patch(color='b', label='Spiculated')
k_patch = mpatches.Patch(color='k', label = 'Obscured')
y_patch = mpatches.Patch(color='y', label='Unknown Type')
m_patch = mpatches.Patch(color='m', label='Micro-lobulated')
plt.legend(handles=[r_patch, g_patch, b_patch, k_patch, y_patch, m_patch])
plt.title('Mass Margin Type Among the Sample Population')
plt.xlabel('Mass Margin')
plt.show()

# In[] Exploring the Data - Mode for Margin
density_mode = breast['density'].mode()
## 3 - this is equivalent to 'low'

# In[] Visualising the Data - Bar Chart for Density
breast['density'].value_counts().plot(kind='bar', colors='rgbyk')
r_patch = mpatches.Patch(color='r', label='Low Density')
g_patch = mpatches.Patch(color='g', label='Unknown Density')
b_patch = mpatches.Patch(color='b', label='Iso Density')
y_patch = mpatches.Patch(color='y', label='High Density')
k_patch = mpatches.Patch(color='k', label='Fat-Containing')
plt.legend(handles=[r_patch, g_patch, b_patch, y_patch, k_patch])
plt.title('The proportion of Various Mass Densities Among the Sample Population')
plt.xlabel('Mass Density Type')
plt.show()

# In[] Exploring the Data - Mode for Severity
severity_mode = breast['density'].mode()
## 0 - this is equivalent to 'benign'

# In[] Visualising the Data - Pie Chart for Severity
breast['severity'].value_counts().plot(kind='pie', autopct='%.2f', colors=['green', 'blue'])
green_patch = mpatches.Patch(color='green', label='Benign')
blue_patch = mpatches.Patch(color='blue', label='Malignant')
plt.legend(handles=[green_patch, blue_patch])
plt.title('Severity of the Present Mass')
plt.xlabel('Mass Severity')
plt.ylabel('Percentage Observed within Sample Population')
plt.show()

# In[] Age vs BIRADS boxplot
breast.boxplot(column='age', by='BIRADS')
plt.xlabel('BIRADS')
plt.ylabel('Age (years)')
plt.show()

# In[] Age vs Shape boxplot
breast.boxplot(column='age', by='shape')
plt.xlabel('Shape Type')
plt.ylabel('Age (years)')
plt.show()

# In[] Age vs Margin boxplot
breast.boxplot(column='age', by='margin')
plt.xlabel('Margin Type')
plt.ylabel('Age (years)')
plt.show()

# In[] Age vs Density boxplot
breast.boxplot(column='age', by='density')
plt.xlabel('Density Type')
plt.ylabel('Age (years)')
plt.show()

# In[] Age vs Severity boxplot
breast.boxplot(column='age', by='severity')
plt.xlabel('Mass Severity')
plt.ylabel('Age (years)')
plt.show()

# In[] Severity vs margin boxplot
breast.boxplot(column='margin', by='severity')
plt.ylim(-0.5,6)
plt.xlabel('Severity of Mass')
plt.ylabel('Margin Type')
plt.show()

# In[] Severity vs shape boxplot
breast.boxplot(column='shape', by='severity')
plt.ylim(-0.5,5)
plt.xlabel('Severity of Mass')
plt.ylabel('Shape Type')
plt.show()

# In[] Severity vs BIRADS boxplot
breast.boxplot(column='BIRADS', by='severity')
plt.ylim(-0.5,7)
plt.xlabel('Severity of Mass')
plt.ylabel('BIRADS Score')
plt.show()

# In[] Shape vs Density boxplot
breast.boxplot(column='shape', by='density')
plt.ylim(-0.5,5)
plt.xlabel('Mass Density')
plt.ylabel('Mass Shape')
plt.show()

# In[] Density vs Severity boxplot
breast.boxplot(column='density', by='severity')
plt.ylim(-0.5,5)
plt.xlabel('Severity of Mass')
plt.ylabel('Density Type')
plt.show()

# In[] Generating Train & Test Sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast, breast.severity, test_size = 0.20, random_state=4)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# In[] Data Modelling - K-NN Creating the Model
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(2)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)

# In[] Data Modelling - K-NN Assessing the Model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted)
print cm
from sklearn.metrics import classification_report
print classification_report (y_test, predicted)

# In[] Data Modelling - Decision Tree Classifier Creating the Model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=6, max_features=2)
fit = clf.fit(X_train, y_train)
y_pre = fit.predict(X_test)   
y_pre
y_pre.shape

# In[] Data Modelling - Decision Tree Classifier Assessing the Model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pre)
print cm
from sklearn.metrics import classification_report
print classification_report(y_test, y_pre)

# In[] Visualisation of Decision Tree
from sklearn import tree









