#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:44:27 2017

@author: alistairgj
"""
URL - http://archive.ics.uci.edu/ml/datasets/mammographic+mass

# In[] Importing the data analysis libraries

import pandas as pd
import numpy as np
import urllib2
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
breast['BIRADS'].value_counts()

# In[] Cleaning the data - Age
breast['age'].value_counts()
mask_unknown = '?'
breast.loc[breast['age'] == mask_unknown, 'age'] = 60
breast['age'] = breast['age'].str.strip()
breast['age'].value_counts()
## Median breast cancer age of 60 years was used to fill missing

# In[] Cleaning the data - Shape
breast['shape'].value_counts()
mask_unknown = '?'
breast.loc[breast['shape'] == mask_unknown, 'shape'] = 0
breast['shape'] = breast['shape'].str.strip()
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

# In[] Cleaning the data - Density

######Cleaning severity:already fine and set to int

breast['severity'].value_counts()

##########Changing some of out Nan values that have arisen------------------
#Check for Nan values

breast.isnull().any()
#If any true, fill Nan with appropriate value

#in BIRADS replace Nan with 0

breast['BIRADS'].fillna(0, inplace=True)

#in age replace Nan with 60

breast['age'].fillna(60, inplace=True)

#in shape replace Nan with 0

breast['shape'].fillna(0, inplace=True)

breast.isnull().any()

#------------------------changing the data types-------------------------
breast.dtypes

breast['BIRADS'] = breast['BIRADS'].astype(int)

#age could only change to float for some reason
#worked for int??

breast['age'] = breast['age'].astype(int)

breast['shape'] = breast['shape'].astype(int)

breast['margin'] = breast['margin'].astype(int)

breast['density'] = breast['density'].astype(int)

#severity already set to int

print breast

breast.dtypes



# *************************** Task 2: Data Exploration ******************************

# Write your code immediately below:

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#------------------------------
## BIRADS

breast['BIRADS'].mode()
## 4

BREAST['BIRADS'].value_counts()

## BAR GRAPH FOR BIRADS

breast['BIRADS'].value_counts().plot(kind='bar')

plt.ylabel('Frequency')

plt.xlabel('Ranking')

plt.title('Mass BI-RADS Assessment Scores')

plt.show()

#------------------------------
## AGE 

breast['age'].mean()
## 55.51

breast['age'].min()
## 18

breast['age'].max()
## 96

breast['age'].median()
# 57.0

breast['age'].mode()
# 59

breast['age'].value_counts()

## HISTOGRAM FOR AGE 

breast['age'].plot(kind='hist', bins=20, colors=['skyblue'])

plt.title('Distribution of Age Amongst the Sample Population')

plt.xlabel('Age (years)')

plt.show()

#-------------------------------

## SHAPE

breast['shape'].mode()
## 4
##equivalent to ''irregular''

breast['shape'].value_counts()

## SHAPE PIE CHART 

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

#------------------------------

## MARGIN

breast['margin'].mode()
## 1
# equivalent to margin being 'circumscribed'

breast['margin'].value_counts()

## MARGIN BAR CHART

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

#--------------------------------

## DENSITY

breast['density'].value_counts()

breast['density'].mode()
## 3
#equivalent to 'low'

## DENSITY BAR GRAPH 

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


#----------------------------------

## SEVERITY

breast['severity'].value_counts()

breast['severity'].mode()
## 0
#equivalent to 'benign'

## SEVERITY PIE CHART

breast['severity'].value_counts().plot(kind='pie', autopct='%.2f', colors=['green', 'blue'])

green_patch = mpatches.Patch(color='green', label='Benign')

blue_patch = mpatches.Patch(color='blue', label='Malignant')

plt.legend(handles=[green_patch, blue_patch])

plt.title('Severity of the Present Mass')

plt.xlabel('Mass Severity')

plt.ylabel('Percentage Observed within Sample Population')

plt.show()

#-----------------------------

## RELATIONSHIPS BETWEEN ATTRIBUTES

#-------------------------------

## Age vs BIRADS BOXPLOT

breast.boxplot(column='age', by='BIRADS')

plt.xlabel('BIRADS')

plt.ylabel('Age (years)')

plt.show()

#-----------------------

## Age vs Shape

breast.boxplot(column='age', by='shape')

plt.xlabel('Shape Type')

plt.ylabel('Age (years)')

plt.show()

#-----------------------

## Age vs Margin

breast.boxplot(column='age', by='margin')

plt.xlabel('Margin Type')

plt.ylabel('Age (years)')

plt.show()

#------------------------

## Age vs Density

breast.boxplot(column='age', by='density')

plt.xlabel('Density Type')

plt.ylabel('Age (years)')

plt.show()

#--------------------------

## Age vs Severity 

breast.boxplot(column='age', by='severity')

plt.xlabel('Mass Severity')

plt.ylabel('Age (years)')

plt.show()

#---------------------------

## severity vs margin

breast.boxplot(column='margin', by='severity')

plt.ylim(-0.5,6)

plt.xlabel('Severity of Mass')

plt.ylabel('Margin Type')

plt.show()

#--------------------------

## severity vs shape

breast.boxplot(column='shape', by='severity')

plt.ylim(-0.5,5)

plt.xlabel('Severity of Mass')

plt.ylabel('Shape Type')

plt.show()

#-------------------------

## severity vs BIRADS

breast.boxplot(column='BIRADS', by='severity')

plt.ylim(-0.5,7)

plt.xlabel('Severity of Mass')

plt.ylabel('BIRADS Score')

plt.show()

#-------------------------

## SHAPE vs DENSITY

breast.boxplot(column='shape', by='density')

plt.ylim(-0.5,5)

plt.xlabel('Mass Density')

plt.ylabel('Mass Shape')

plt.show()

#--------------------------

## Density vs Severity

breast.boxplot(column='density', by='severity')

plt.ylim(-0.5,5)

plt.xlabel('Severity of Mass')

plt.ylabel('Density Type')

plt.show()




# *************************** Task 3: Data Modelling ******************************
# Write your code immediately below:



##Generating TrainTest Set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(breast, breast.severity, test_size = 0.20, random_state=4)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

##-------------------

## DATA MODELLING - K-NEAREST NEIGHBOUR 
## explore different k values

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(2)

fit = clf.fit(X_train, y_train)

predicted = fit.predict(X_test)

predicted

predicted.shape

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,predicted)

print cm

from sklearn.metrics import classification_report

print classification_report (y_test, predicted)

#----------------------------------------------------------------------------

## DATA MODELLING - DECISION TREE CLASSIFIER

###---------------SETTING CONSTRAINTS ON DECISION TREE-----------------------

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=6, max_features=2)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)   

y_pre

y_pre.shape

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pre)

print cm

from sklearn.metrics import classification_report

print classification_report(y_test, y_pre)


####### Visualisation of Decision Tree

from sklearn import tree

#Define a location to put the code for the decision tree - I chose my H Drive, in
#the folder data_science (in RMIT mydesktop)

with open ("HDrive/data_science/breast.txt", 'w') as f:
	f = tree.export_graphviz (clf, out_file=f, feature_names = breast.columns)

#I then accessed the breast.txt file in the location H:/data_science
#Copied and pasted the code generated for the decision tree into the website http://webgraphvis.com
#took a screenshot of the decision tree

##-------------------------------------------------------------------------------

digraph Tree {
node [shape=box] ;
0 [label="margin <= 1.5\ngini = 0.4982\nsamples = 768\nvalue = [407, 361]"] ;
1 [label="age <= 64.5\ngini = 0.2268\nsamples = 322\nvalue = [280, 42]"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
2 [label="severity <= 0.5\ngini = 0.1829\nsamples = 275\nvalue = [247, 28]"] ;
1 -> 2 ;
3 [label="gini = 0.0\nsamples = 247\nvalue = [247, 0]"] ;
2 -> 3 ;
4 [label="gini = 0.0\nsamples = 28\nvalue = [0, 28]"] ;
2 -> 4 ;
5 [label="age <= 70.5\ngini = 0.4183\nsamples = 47\nvalue = [33, 14]"] ;
1 -> 5 ;
6 [label="density <= 2.5\ngini = 0.375\nsamples = 28\nvalue = [21, 7]"] ;
5 -> 6 ;
7 [label="gini = 0.32\nsamples = 5\nvalue = [4, 1]"] ;
6 -> 7 ;
8 [label="age <= 65.5\ngini = 0.3856\nsamples = 23\nvalue = [17, 6]"] ;
6 -> 8 ;
9 [label="gini = 0.48\nsamples = 5\nvalue = [3, 2]"] ;
8 -> 9 ;
10 [label="age <= 66.5\ngini = 0.3457\nsamples = 18\nvalue = [14, 4]"] ;
8 -> 10 ;
11 [label="gini = 0.32\nsamples = 5\nvalue = [4, 1]"] ;
10 -> 11 ;
12 [label="gini = 0.355\nsamples = 13\nvalue = [10, 3]"] ;
10 -> 12 ;
13 [label="shape <= 2.5\ngini = 0.4654\nsamples = 19\nvalue = [12, 7]"] ;
5 -> 13 ;
14 [label="age <= 72.5\ngini = 0.4898\nsamples = 14\nvalue = [8, 6]"] ;
13 -> 14 ;
15 [label="gini = 0.48\nsamples = 5\nvalue = [2, 3]"] ;
14 -> 15 ;
16 [label="gini = 0.4444\nsamples = 9\nvalue = [6, 3]"] ;
14 -> 16 ;
17 [label="gini = 0.32\nsamples = 5\nvalue = [4, 1]"] ;
13 -> 17 ;
18 [label="margin <= 4.5\ngini = 0.4073\nsamples = 446\nvalue = [127, 319]"] ;
0 -> 18 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
19 [label="shape <= 2.5\ngini = 0.4377\nsamples = 340\nvalue = [110, 230]"] ;
18 -> 19 ;
20 [label="BIRADS <= 4.5\ngini = 0.4558\nsamples = 74\nvalue = [48, 26]"] ;
19 -> 20 ;
21 [label="severity <= 0.5\ngini = 0.3062\nsamples = 53\nvalue = [43, 10]"] ;
20 -> 21 ;
22 [label="gini = 0.0\nsamples = 43\nvalue = [43, 0]"] ;
21 -> 22 ;
23 [label="gini = 0.0\nsamples = 10\nvalue = [0, 10]"] ;
21 -> 23 ;
24 [label="gini = 0.3628\nsamples = 21\nvalue = [5, 16]"] ;
20 -> 24 ;
25 [label="severity <= 0.5\ngini = 0.3575\nsamples = 266\nvalue = [62, 204]"] ;
19 -> 25 ;
26 [label="gini = 0.0\nsamples = 62\nvalue = [62, 0]"] ;
25 -> 26 ;
27 [label="gini = 0.0\nsamples = 204\nvalue = [0, 204]"] ;
25 -> 27 ;
28 [label="shape <= 3.5\ngini = 0.2693\nsamples = 106\nvalue = [17, 89]"] ;
18 -> 28 ;
29 [label="BIRADS <= 4.5\ngini = 0.1884\nsamples = 19\nvalue = [2, 17]"] ;
28 -> 29 ;
30 [label="gini = 0.48\nsamples = 5\nvalue = [2, 3]"] ;
29 -> 30 ;
31 [label="gini = 0.0\nsamples = 14\nvalue = [0, 14]"] ;
29 -> 31 ;
32 [label="severity <= 0.5\ngini = 0.2854\nsamples = 87\nvalue = [15, 72]"] ;
28 -> 32 ;
33 [label="gini = 0.0\nsamples = 15\nvalue = [15, 0]"] ;
32 -> 33 ;
34 [label="gini = 0.0\nsamples = 72\nvalue = [0, 72]"] ;
32 -> 34 ;
}


# *************************** Optional Extension ******************************
# Write your code immediately below:






# ******************************************************************************
# **************************** End of Document *********************************
# ******************************************************************************