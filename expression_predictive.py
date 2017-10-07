# In[1] Importing the data analysis library pandas
import pandas as pd

# In[2] Importing scientific computing library numpy
import numpy as np

# In[3] Importing the python plotting library matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

# In[4] Importing the library for opening URLs
import urllib2

# In[5] Creating string abbreviation of URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"

# In[6]
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
# Write this dataframe to a CSV file for later use
targetProteins.to_csv("finalData.csv")

# Create a dataframe of the average value of each protein for each mouse
mouseRows = []
for mouseID in miceIDs:
    row = {}
    row['MouseID'] = mouseID
    for proteinName in proteinNames:
        row[proteinName] = targetProteins[targetProteins['MouseID'].str.contains(mouseID)][proteinName].mean()

    row['Genotype'] = targetProteins[targetProteins['MouseID'].str.contains(mouseID)]['Genotype'].iloc[0]
    row['Treatment'] = targetProteins[targetProteins['MouseID'].str.contains(mouseID)]['Treatment'].iloc[0]
    row['Behavior'] = targetProteins[targetProteins['MouseID'].str.contains(mouseID)]['Behavior'].iloc[0]
    row['Class'] = targetProteins[targetProteins['MouseID'].str.contains(mouseID)]['Class'].iloc[0]
    mouseRows.append(row)

cols = proteinNames + ['Genotype', 'Treatment', 'Behavior', 'Class']

miceDF = pd.DataFrame(mouseRows, index=miceIDs, columns=cols)

print miceDF

# Write this dataframe to a CSV file for later use
miceDF.to_csv('miceAverages.csv')

    
# In[]



# In[17]:

scatterProteins2 = miceDF[['BRAF_N', 'pERK_N', 'S6_N', 'pGSK3B_N', 'CaNA_N', 'CDK5_N', 'pNUMB_N', 'DYRK1A_N', 'ITSN1_N', 'SOD1_N', 'GFAP_N']]
scatter_matrix(scatterProteins2, alpha=0.2,figsize=(16,16),diagonal='hist')
plt.suptitle('Cleaned Data with Average per Mouse per Protein Scatter Matrix')
plt.show()
