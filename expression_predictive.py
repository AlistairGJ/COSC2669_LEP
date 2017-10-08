


# In[1] Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import urllib2
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import os

# In[5] Creating string abbreviation of URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
proteinExpression = pd.read_excel(urllib2.urlopen(url), headers=0)

# In[7] Checking the column (attribute) names - exploring data with a list
columnIDlist = proteinExpression.columns.tolist()

# In[8] Confirming the data types (exploration) - series created
nativeDataTypes = proteinExpression.dtypes

# In[9] Value counts of data types
countDataTypes = nativeDataTypes.value_counts()

# In[10] Checking the value counts of categorical columns
proteinExpression[['Genotype', 'Treatment', 'Behavior', 'class']].apply(pd.Series.value_counts)

# In[10] Rename class column with uppercase 
proteinExpression.rename(columns = {'class': 'Class'}, inplace = True)

# In[11] Create new data set with desired proteins
targetProteins = proteinExpression[['MouseID', 'BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 
                        'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 
                        'GFAP_N', 'Genotype', 'Treatment', 'Behavior', 'Class']]

# In[12] Stats check
targetProteinsStats = targetProteins.describe()

# In[13] Checking for missing values
missingValueCheck = targetProteins[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 
                        'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 
                        'ITSN1_N', 'SOD1_N', 'GFAP_N']].isnull()

# In[14] Missing value count
missingValueList = missingValueCheck.apply(pd.Series.value_counts)

# In[15] Housekeeping
pd.options.mode.chained_assignment = None

# In[16 - 24]: Removing string categories
targetProteins['Genotype'].replace('Control', '1', inplace=True)
targetProteins['Genotype'].replace('Ts65Dn', '0', inplace=True)
targetProteins['Genotype'] = targetProteins['Genotype'].astype(int)
targetProteins['Behavior'].replace('C/S', '1', inplace=True)
targetProteins['Behavior'].replace('S/C', '0', inplace=True)
targetProteins['Behavior'] = targetProteins['Behavior'].astype(int)
targetProteins['Treatment'].replace('Saline', '1', inplace=True)
targetProteins['Treatment'].replace('Memantine', '0', inplace=True)
targetProteins['Treatment'] = targetProteins['Treatment'].astype(int)

# In[17] Confirming alteration
targetProteins[['Behavior', 'Genotype', 'Treatment']].apply(pd.Series.value_counts)

# In[18] Converting binary to unique identifying numbers
def change_class(row):
    row['Class'] = row['Genotype'] * 4 + row['Behavior'] * 2 + row['Treatment']
    return row

# In[19] Applying above change to target proteins data
targetProteins = targetProteins.apply(change_class, axis=1)

# In[20]: Checking value counts
targetProteins['Class'].value_counts()

# In[] - Will give back a uniform distribution from 0 till 1
targetProteinsBoxPlot = pd.DataFrame(np.random.rand(1080, 11), columns=['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 
                        'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N'])
targetProteinsBoxPlot.boxplot(figsize=(4,4), vert=False)

# In[] - Will give back a normalized (mean = 0) box plot with outliers
targetProteinsBoxPlot = pd.DataFrame(np.random.randn(1080, 11), columns=['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 
                        'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N'])
targetProteinsBoxPlot.boxplot(figsize=(4,4), vert=False)

# In[]    
from pandas.tools.plotting import scatter_matrix
scatterProteins = targetProteins[['MouseID', 'BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N']]
scatter_matrix(scatterProteins, alpha=0.2,figsize=(16,16),diagonal='hist')
plt.show()

# In[]:
proteinNames = ['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 
                'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N']
miceIDs = proteinExpression['MouseID'].str.split('_').apply(pd.Series, 1)[0].unique()
MakeMouseID = proteinExpression['MouseID'].str.split('_').apply(pd.Series, 1)[0].unique()

# In[TEST]:
proteinRows = []
for proteinName in proteinNames:
    count = targetProteins[proteinName].count()
    mean = targetProteins[proteinName].mean()
    sd = targetProteins[proteinName].std()
    minusThreeSD = mean - (3 * sd)
    minusTwoSD = mean - (2 * sd)
    twoSD = mean + (2 * sd)
    threeSD = mean + (3 * sd)
    outliers = targetProteins.query(proteinName + ' < ' + str(minusThreeSD) + ' | ' + proteinName + ' > ' + str(threeSD))

    row = {'Protein': proteinName,'Count': count,'Mean': mean,'SD': sd,
           '-3SD': minusThreeSD,'-2SD': minusTwoSD,'+2SD': twoSD,
           '+3SD': threeSD,'Outliers': outliers[proteinName].count()}
    proteinRows.append(row)

nnPctRangeDF = pd.DataFrame(proteinRows, index=proteinNames, 
                            columns=['Count', 'Mean', 'SD', '-3SD', 
                                     '-2SD', '+2SD', '+3SD', 'Outliers'])
#nnPctRangeDF shows us the number of outliers count for each protein
    
# In[]
outlierMiceRows = []
for proteinName in proteinNames:

    mean = targetProteins[proteinName].mean()
    sd = targetProteins[proteinName].std()
    minusThreeSD = mean - (3 * sd)
    threeSD = mean + (3 * sd)
    outliers = targetProteins.query(proteinName + ' < ' + str(minusThreeSD) + ' | ' + proteinName + ' > ' + str(threeSD))

    if outliers.empty:
        row = {'Protein': proteinName,'MouseID': '-','# Instances': '-','Genotype': '-',
           'Treatment': '-','Behavior': '-','Class': '-'}
        outlierMiceRows.append(row)
    else:
        for mouseID in miceIDs:
            mouseOutlierRows = outliers[outliers['MouseID'].str.contains(mouseID)]
            if not mouseOutlierRows.empty:
                    row = {'Protein': proteinName,'MouseID': mouseID,
                           '# Instances': len(mouseOutlierRows),
                           'Genotype': mouseOutlierRows['Genotype'].iloc[0],
                           'Treatment': mouseOutlierRows['Treatment'].iloc[0],
                           'Behavior': mouseOutlierRows['Behavior'].iloc[0],
                           'Class': mouseOutlierRows['Class'].iloc[0]}
                    outlierMiceRows.append(row)

outliersDF = pd.DataFrame(outlierMiceRows, 
                          columns=['Protein', 'MouseID','# Instances', 
                                   'Genotype','Treatment', 'Behavior', 'Class'])

# In[] Stripping all outliers back to NaN
def make_nans(row):
    for proteinName in proteinNames:
        mean = targetProteins[proteinName].mean()
        sd = targetProteins[proteinName].std()
        minusThreeSD = mean - (3 * sd)
        threeSD = mean + (3 * sd)
    
        if row[proteinName] < minusThreeSD or row[proteinName] > threeSD:
            row[proteinName] = None
        return row
targetProteins = targetProteins.apply(make_nans, axis=1)

# In[] Removal of mouse 3484_n
targetProteins = targetProteins[~targetProteins['MouseID'].str.contains('3484')]
indexOfMouse = np.where(miceIDs=='3484')[0]
miceIDs = np.delete(miceIDs, indexOfMouse)

# In[] Convert all remaining NaNs to the average value of that protein for that class
def make_averages(row):
    for proteinName in proteinNames:
        if np.isnan(row[proteinName]):
            average = targetProteins[targetProteins.Class == row['Class']][proteinName].mean()
            row[proteinName] = average
    return row
targetProteins = targetProteins.apply(make_averages, axis=1)

# In[]
scatterProteins = targetProteins[['MouseID', 'BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N']]
scatter_matrix(scatterProteins, alpha=0.2,figsize=(16,16),diagonal='hist')
plt.show()

# In[]
targetProteins.describe()

# In[]
targetProteins['Genotype'].value_counts()

# In[]
targetProteins['Treatment'].value_counts()

# In[]
targetProteins['Behavior'].value_counts()

# In[]
targetProteins['Class'].value_counts()

    
# In[]



# In[]:

#Task 3: Data Modelling (Classification)

#Classification 1: K Nearest Neighbor

    
# In[]

#Using SOD1, Genotype & Treatment to predict Behavior
#Function for classification
targetProteins_SOD1 = targetProteins[['SOD1_N', 'Genotype', 'Treatment']]
targetProteins_SOD1.dtypes
targetProteins_SOD1.describe()
X_train, X_test, y_train, y_test = train_test_split(targetProteins_SOD1, targetProteins['Behavior'], test_size=0.4)
X_train.shape
y_train.shape

# In[] Analysis: 5 neighbours - Using SOD1, Genotype & Treatment to predict Behavior
clf = KNeighborsClassifier(5)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[] Analysis: 2 neighbours - Using SOD1, Genotype & Treatment to predict Behavior
clf = KNeighborsClassifier(2)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "2 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[] Analysis: 8 neighbours - Using SOD1, Genotype & Treatment to predict Behavior
clf = KNeighborsClassifier(8)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[] Regression - Input SOD1_N, S6_N, GFAP_N with Output CaNA_N
targetProteins_regression = targetProteins[['SOD1_N', 'S6_N', 'GFAP_N']]
X_train, X_test, y_train, y_test = train_test_split(targetProteins_regression, targetProteins['CaNA_N'], test_size=0.2)
X_train.shape
y_train.shape

# In[]
regressor = DecisionTreeRegressor()
fit = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse_CaNA_N = mean_squared_error(y_test, y_pred)
print mse_CaNA_N
print classification_report (y_test, mse_CaNA_N)

# In[] Regression - Input pNUMB_N, S6_N with Output CaNA_N
targetProteins_regression = targetProteins[['pNUMB_N', 'S6_N']]
X_train, X_test, y_train, y_test = train_test_split(targetProteins_regression, targetProteins['CaNA_N'], test_size=0.2)
X_train.shape
y_train.shape

# In[]
regressor = DecisionTreeRegressor()
fit = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse_CaNA_N = mean_squared_error(y_test, y_pred)
print mse_CaNA_N
print classification_report (y_test, mse_CaNA_N)


# In[] Regression - Input S6_N with Output CaNA_N
targetProteins_regression = targetProteins[['S6_N']]
X_train, X_test, y_train, y_test = train_test_split(targetProteins_regression, targetProteins['CaNA_N'], test_size=0.2)
X_train.shape
y_train.shape

# In[]
regressor = DecisionTreeRegressor()
fit = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse_CaNA_N = mean_squared_error(y_test, y_pred)
print mse_CaNA_N
print classification_report (y_test, mse_CaNA_N)

# In[] Regression - Input S6_N with Output CaNA_N
targetProteins_regression = targetProteins[['S6_N', 'pGSK3B_N', 'CDK5_N', 'pGSK3B_N']]
X_train, X_test, y_train, y_test = train_test_split(targetProteins_regression, targetProteins['CaNA_N'], test_size=0.2)
X_train.shape
y_train.shape

# In[]
regressor = DecisionTreeRegressor()
fit = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse_CaNA_N_NEW = mean_squared_error(y_test, y_pred)
print mse_CaNA_N_NEW
print classification_report (y_test, mse_CaNA_N_NEW)



# In[] Using ITSN1, Genotype & Treatment to predict Behavior
targetProteins_ITSN1 = targetProteins[['ITSN1_N', 'Genotype', 'Treatment']]
X_train, X_test, y_train, y_test = train_test_split(targetProteins_ITSN1, targetProteins['Behavior'], test_size=0.2)
X_train.shape
y_train.shape

# In[]

#Analysis: 5 neighbours
clf = KNeighborsClassifier(5)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 2 neighbours
clf = KNeighborsClassifier(2)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "2 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 8 neighbours
clf = KNeighborsClassifier(8)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 5 neighbours, distance weighted
clf = KNeighborsClassifier(5, weights='distance')
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours, distance weighted predicted: "
print cm
print classification_report (y_test, predicted)

# In[]
#Analysis: 8 neighbours, distance weighted
clf = KNeighborsClassifier(8, weights='distance')
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours, distance weighted predicted: "
print cm
print classification_report (y_test, predicted)

# In[]
#Analysis: 5 neighbours, distance weighted, p=1
clf = KNeighborsClassifier(5, weights='distance', p=1)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours, distance weighted predicted, p=1: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 8 neighbours, distance weighted, p=1
clf = KNeighborsClassifier(8, weights='distance', p=1)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours, distance weighted predicted, p=1: "
print cm
print classification_report (y_test, predicted)

# In[]

#Use all protein expression levels to clasify class
targetProteins_class = targetProteins[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N']]
X_train, X_test, y_train, y_test = train_test_split(targetProteins_class, targetProteins['Class'], test_size=0.2)
print X_train.shape
print y_train.shape

# In[]

#Analysis: 5 neighbours
clf = KNeighborsClassifier(5)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 2 neighbours
clf = KNeighborsClassifier(2)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "2 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 8 neighbours
clf = KNeighborsClassifier(8)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours predicted: "
print cm
print classification_report (y_test, predicted)

# In[]


#Analysis: 5 neighbours, distance weighted
clf = KNeighborsClassifier(5, weights='distance')
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours, distance weighted predicted: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 8 neighbours, distance weighted
clf = KNeighborsClassifier(8, weights='distance')
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours, distance weighted predicted: "
print cm
print classification_report (y_test, predicted)

# In[]


#Analysis: 5 neighbours, distance weighted, p=1
clf = KNeighborsClassifier(5, weights='distance', p=1)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "5 neighbours, distance weighted predicted, p=1: "
print cm
print classification_report (y_test, predicted)

# In[]

#Analysis: 8 neighbours, distance weighted, p=1
clf = KNeighborsClassifier(8, weights='distance', p=1)
fit = clf.fit(X_train, y_train)
predicted = fit.predict(X_test)
cm = confusion_matrix(y_test, predicted)
print "8 neighbours, distance weighted predicted, p=1: "
print cm
print classification_report (y_test, predicted)

# In[]

#Classification 2: Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
import os

import pandas as pd

# In[] 

df_class = targetProteins[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N', 'Class']]

y = df_class.pop('Class')
X = df_class

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#X_train, X_test, y_train, y_test = train_test_split(proteins, proteins, random_state = 0)
clf = DecisionTreeClassifier()
fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

print "Class"
print cm
print classification_report(y_test, y_pre)


# In[]

print "Class"
print cm
print classification_report(y_test, y_pre)
with open("class_tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N',
                                            'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N'], class_names="01234567",
                             filled=True, rounded=True, special_characters=True)
os.system("dot class_tree.dot -o class_tree.png -Tpng")

####

# In[]

df_genotype = targetProteins[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N', 'Genotype']]

y = df_genotype.pop('Genotype')
X = df_genotype

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#X_train, X_test, y_train, y_test = train_test_split(proteins, proteins, random_state = 0)
clf = DecisionTreeClassifier()
fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

print "Genotype"
print cm
print classification_report(y_test, y_pre)

# In[]
with open("genotype_tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N',
                                            'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N'], class_names="01",
                             filled=True, rounded=True, special_characters=True)
os.system("dot genotype_tree.dot -o genotype_tree.png -Tpng")


####

df_behavior = df[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N', 'Behavior']]

y = df_behavior.pop('Behavior')
X = df_behavior

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#X_train, X_test, y_train, y_test = train_test_split(proteins, proteins, random_state = 0)
clf = DecisionTreeClassifier()
fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

print "Behavior"
print cm
print classification_report(y_test, y_pre)
with open("behavior_tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N',
                                            'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N'], class_names="01",
                             filled=True, rounded=True, special_characters=True)
os.system("dot behavior_tree.dot -o behavior_tree.png -Tpng")


####

df_treatment = df[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N', 'Treatment']]

y = df_treatment.pop('Treatment')
X = df_treatment

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#X_train, X_test, y_train, y_test = train_test_split(proteins, proteins, random_state = 0)
clf = DecisionTreeClassifier()
fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

print "Treatment"
print cm
print classification_report(y_test, y_pre)
with open("treatment_tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N',
                                            'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N'],
                             class_names=["Mamantine","Saline"],
                             filled=True, rounded=True, special_characters=True)
os.system("dot treatment_tree.dot -o treatment_tree.png -Tpng")

####

df_class_braf = df[['BRAF_N','Class']]

y = df_class_braf.pop('Class')
X = df_class_braf

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#X_train, X_test, y_train, y_test = train_test_split(proteins, proteins, random_state = 0)
clf = DecisionTreeClassifier()
fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

print "Class by BRAF"
print cm
print classification_report(y_test, y_pre)
with open("classbybraf_tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['BRAF_N'], class_names="01234567",
                             filled=True, rounded=True, special_characters=True)
os.system("dot classbybraf_tree.dot -o classbybraf_tree.png -Tpng")




































