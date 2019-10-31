#!/usr/bin/env python
# coding: utf-8

# ## WIFI LOCATIONING

# ### Topic:
# 
# Evaluating the application of machine learning techniques to the problem of indoor locationing via wifi fingerprinting.
# 
# Automatic user localization consists of estimating the position of the user  by using an electronic device, usually a mobile phone. While outdoor localisation can be done by using GPS, indoor localisation is still a technical challenge.
# 
# Localising an electronic device and its user by using the signal strength in connection to Wireless Access Points (WAPs) is one of the innovative answers to this challenge. The goal of this project is to explore the accuracy of this concept to find the exact location of a certain user.

# ### Importing Libraries

# In[1866]:


#import libraries

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data

# In[1867]:


#loading training and testing Data Sets

Wifi_train = pd.read_excel('trainingData.xlsx')

Wifi_validation = pd.read_excel('validationData.xlsx')

raw_train_data = Wifi_train.copy()
raw_valid_data = Wifi_validation.copy()


# ### Initial Data Exploration

# In[1868]:


Wifi_train.head()


# ### Attribute Information:
# 
# 
# 
# **Attribute 001 (WAP001)**: Intensity value for WAP001. Negative integer values from -104 to 0 and +100. Positive value 100 used if WAP001 was not detected.
# ....
# 
# **Attribute 520 (WAP520)**: Intensity value for WAP520. Negative integer values from -104 to 0 and +100. Positive Vvalue 100 used if WAP520 was not detected.
# 
# **Attribute 521 (Longitude)**: Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000
# 
# **Attribute 522 (Latitude)**: Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018.
# 
# **Attribute 523 (Floor)**: Altitude in floors inside the building. Integer values from 0 to 4.
# 
# **Attribute 524 (BuildingID)**: ID to identify the building. Measures were taken in three different buildings. Categorical integer values from 0 to 2.
# 
# **Attribute 525 (SpaceID)**: Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken. Categorical integer values.
# 
# **Attribute 526 (RelativePosition)**: Relative position with respect to the Space (1 - Inside, 2 - Outside in Front of the door). Categorical integer values.
# 
# **Attribute 527 (UserID**): User identifier (see below). Categorical integer values.
# 
# **Attribute 528 (PhoneID)**: Android device identifier (see below). Categorical integer values.
# 
# **Attribute 529 (Timestamp)**: UNIX Time when the capture was taken. Integer value. 

# In[1869]:


Wifi_train.info()

pd.set_option('display.float_format', lambda x: '%.6f' % x)


# In[1870]:


Wifi_train.describe()


# In[1871]:


#are there any missing values in the data set?

print(Wifi_train.isnull().values.any())
print(Wifi_validation.isnull().values.any())


# In[1872]:


#weakest signal of all WAP's
print(Wifi_train.iloc[:,0:520].min().min())
print(Wifi_validation.iloc[:,0:520].min().min())


# In[1873]:


#maximum value for signals of WAP's (meaning no signal at all)
Wifi_train.iloc[:,0:520].max().max()


# In[1874]:


Wifi_train.columns


# In[1875]:


print(Wifi_train.shape)
print(Wifi_validation.shape)


# In[1]:


#correlation between features except WAP's
corr = Wifi_train.iloc[:,520:529].corr()


# In[1877]:


# Heatmap for correlation between features (except WAPs)

fig, ax = plt.subplots(1, 1, figsize = (16, 6))

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[1878]:


sns.pairplot(Wifi_train[['LONGITUDE','LATITUDE','FLOOR','BUILDINGID','RELATIVEPOSITION','SPACEID','USERID','PHONEID']])


# In[1879]:


fig, ax = plt.subplots(1, 1, figsize = (12, 6))

sns.scatterplot(Wifi_train['LONGITUDE'],Wifi_train['LATITUDE'],hue=Wifi_train['BUILDINGID'],palette='Dark2_r')


# In[1880]:


fig, ax = plt.subplots(1, 1, figsize = (12, 6))

sns.scatterplot(Wifi_train['LONGITUDE'],Wifi_train['LATITUDE'],hue=Wifi_train['FLOOR'])


# In[1881]:


#number of Floors per BuildingID

fig, ax = plt.subplots(1, 1, figsize = (10, 6))

sns.scatterplot(Wifi_train['BUILDINGID'],Wifi_train['FLOOR'],s= 200,marker='.',hue=Wifi_train['BUILDINGID'],palette='Dark2_r',legend=None)
ax.set_xticks(range(0,3))


# In[1882]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6))
h = sns.countplot(x='USERID',data= Wifi_train, palette="Greens_d")


# In[1883]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6))
sns.countplot(x='PHONEID', data= Wifi_train, color='darkblue')


# In[1884]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6))
sns.distplot(list(Wifi_train['SPACEID'] ), kde = False)


# In[1885]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6))
sns.countplot(x='RELATIVEPOSITION', data=Wifi_train, color='g')


# In[1886]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6))
sns.countplot(x='BUILDINGID',data=Wifi_train, hue="FLOOR",palette='Greens_r')


# In[1887]:


# How many fingerprints per User?

Wifi_train['USERID'].value_counts()


# ## Data Preparation

# #change data type from numeric to character for some features
# 
# Wifi_train[['FLOOR','BUILDINGID','RELATIVEPOSITION','SPACEID','USERID','PHONEID']] \
# = Wifi_train[['FLOOR','BUILDINGID','RELATIVEPOSITION','SPACEID','USERID','PHONEID']].astype('category')
# 
# Wifi_validation[['FLOOR','BUILDINGID','RELATIVEPOSITION','SPACEID','USERID','PHONEID']] \
# = Wifi_validation[['FLOOR','BUILDINGID','RELATIVEPOSITION','SPACEID','USERID','PHONEID']].astype('category')

# In[1888]:


#Converting TIMESTAMP into Date Time format

Wifi_train['TIMESTAMP'] = Wifi_train['TIMESTAMP'].apply(datetime.fromtimestamp)
Wifi_validation['TIMESTAMP'] = Wifi_validation['TIMESTAMP'].apply(datetime.fromtimestamp)


# In[1889]:


#converting UTM coordinates to longitude and latitude

#myProj = Proj("+proj=utm +zone=30S, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")


# In[1890]:


#Longitude, Latitude = myProj(Wifi_train['LONGITUDE'].values, Wifi_train['LATITUDE'].values, inverse=True)
#Longitude2, Latitude2 = myProj(Wifi_validation['LONGITUDE'].values, Wifi_validation['LATITUDE'].values, inverse=True)


# In[1891]:


#replacing features Longitude, Latitude with new values

#Wifi_train['LONGITUDE'] = Longitude

#Wifi_train['LATITUDE'] = Latitude

#Wifi_validation['LONGITUDE'] = Longitude2

#Wifi_validation['LATITUDE'] = Latitude2


# In[1892]:


#Sort Data Frame on correct TIMESTAMP order

Wifi_train = Wifi_train.sort_values('TIMESTAMP')
Wifi_validation = Wifi_validation.sort_values('TIMESTAMP')


# ### Duplicates

# In[1893]:


#shape of Data Sets before cleaning process

print(Wifi_train.shape)
print(Wifi_validation.shape)


# In[1894]:


#drop duplicates of wifi fingerprints to avoid redundancy of information

WAP_columns = list(Wifi_train.columns[0:529])

Wifi_train.drop_duplicates(subset= WAP_columns, keep='first',inplace=True)
Wifi_validation.drop_duplicates(subset= WAP_columns, keep='first',inplace=True)

print(Wifi_train.shape)
print(Wifi_validation.shape)


# In[1895]:


#all WAP's in a separated Data Frame
WAPs = Wifi_train.iloc[:,:520]
WAPs2 = Wifi_validation.iloc[:,:520]


# ### New range for signal strength of WAPs

# In[1896]:


#transpose Data Frame to visualize WAPs as independent variables

WAPs_trans = WAPs.transpose()


# In[1897]:


#Detecting suspicious signal strengths/ouliers of WAPs

from itertools import chain

fig, ax = plt.subplots(1, 1, figsize = (10, 5))

plt.scatter( list(chain.from_iterable( [WAPs_trans.index.values.tolist()]*19300)), WAPs_trans,color = 'darkgrey',alpha=0.5)

plt.plot([0,520],[-25,-25],'k-', lw=5, color='red')

plt.xticks(list(range(0,520,20)), rotation='vertical')


# In[1898]:


#adjusting signal strength value=100 (no signal) into plausible range (lower than weakest signal)
print(WAPs.min().min())
print(WAPs2.min().min())


# In[1899]:


#new value for 'no signal'= -105 

WAPs[WAPs==100] = -105
WAPs2[WAPs2==100] = -105

Wifi_train.iloc[:,:520] = WAPs
Wifi_validation.iloc[:,:520] =WAPs2


# In[1900]:


#WAPs with changed 'no signal' of -105 instead if +100

from itertools import chain

fig, ax = plt.subplots(1, 1, figsize = (10, 6))

plt.scatter( list(chain.from_iterable( [WAPs_trans.index.values.tolist()]*19300)), WAPs_trans,color = 'darkgrey',alpha=0.5)

plt.plot([0,520],[-25,-25],'k-', lw=5, color='red')

plt.xticks(list(range(0,520,20)), rotation='vertical')


# ### Removing redundant columns (WAPs)

# In[1901]:


#columns(WAPs) with no signal connection or constant signal don't add any information -> removed

l = Wifi_train.iloc[:,:520].max()==Wifi_train.iloc[:,:520].min()

l = l[l==True]
      
l.describe()

i = l.index


# In[1902]:


#dropping columns 

Wifi_train.drop(i,axis=1, inplace=True)
Wifi_validation.drop(i,axis=1, inplace=True)

WAPs.drop(i, axis=1, inplace=True)


# In[1903]:


#dropping columns which don't add relevant information for the modelling process

Wifi_train.drop(['SPACEID','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1, inplace=True)

#Wifi_validation.drop(['SPACEID','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1, inplace=True)


# In[1904]:


print(Wifi_train.shape)
print(Wifi_validation.shape)


# ### Removing Outliers

# In[1905]:


Wifi_train_col = Wifi_train.copy()
Wifi_validation_col = Wifi_validation.copy()


# In[1906]:


#Outliers with signal strength between 0 and -25 are excluded

Wifi_train = Wifi_train.where(Wifi_train.iloc[:,:-4]<=-25)
Wifi_validation = Wifi_validation.where(Wifi_validation.iloc[:,:-9]<=-25)


# In[1907]:


WAPs = WAPs.where(WAPs <= -25)
WAPs2 = WAPs2.where(WAPs2 <= -25)


# In[1908]:


Wifi_train['LONGITUDE']= Wifi_train_col['LONGITUDE']
Wifi_train['LATITUDE']= Wifi_train_col['LATITUDE']
Wifi_train['BUILDINGID']= Wifi_train_col['BUILDINGID']
Wifi_train['FLOOR']= Wifi_train_col['FLOOR']

Wifi_validation['LONGITUDE']= Wifi_validation_col['LONGITUDE']
Wifi_validation['LATITUDE']= Wifi_validation_col['LATITUDE']
Wifi_validation['BUILDINGID']= Wifi_validation_col['BUILDINGID']
Wifi_validation['FLOOR']= Wifi_validation_col['FLOOR']
Wifi_validation['TIMESTAMP']=Wifi_validation_col['TIMESTAMP']
Wifi_validation['SPACEID']=Wifi_validation_col['SPACEID']
Wifi_validation['RELATIVEPOSITION']=Wifi_validation_col['RELATIVEPOSITION']
Wifi_validation['PHONEID']=Wifi_validation_col['PHONEID']
Wifi_validation['USERID']=Wifi_validation_col['USERID']


# In[1909]:


print(Wifi_train.isnull().sum().sum())
print(Wifi_validation.isnull().sum().sum())


# In[1910]:


print(WAPs.isnull().sum().sum())
print(WAPs2.isnull().sum().sum())


# In[1911]:


Wifi_train = Wifi_train.dropna()
Wifi_validation = Wifi_validation.dropna()
WAPs = WAPs.dropna()
WAPs2 = WAPs2.dropna()


# ### Shape of Clean Data Set

# In[1912]:


#Shape of Data Set before Cleaning and Pre_Processing

print(raw_train_data.shape)
print(raw_valid_data.shape)


# In[1913]:


#Shape of Data Set after Cleaning Process

print(Wifi_train.shape)
print(Wifi_validation.shape)


# In[1914]:


WAPs_trans = WAPs.transpose()

fig, ax = plt.subplots(1, 1, figsize = (18, 9))

plt.scatter( list(chain.from_iterable( [WAPs_trans.index.values.tolist()]*18825)), WAPs_trans,color = 'darkgrey',alpha=0.5)

plt.plot([0,465],[-25,-25],'k-', lw=5, color='red')

plt.xticks(list(range(0,465,20)), rotation='vertical')


# ### Training Model for Predictions

# ## Predicting BUILDINGID 
# 
# ## SVC

# In[1915]:


Wifi_trainc = Wifi_train
Wifi_validationc = Wifi_validation


# In[1916]:


# Classification 
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification


# In[1917]:


# split Data in training and test sets

from sklearn.model_selection import train_test_split


#separate dependent(target) variable and independent variables

target = Wifi_trainc.loc[:, Wifi_trainc.columns == 'BUILDINGID']

df = Wifi_trainc.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)


# In[1918]:


#splitting validation data set

y_validation = Wifi_validationc.loc[:, Wifi_validationc.columns == 'BUILDINGID']

X_validation = Wifi_validationc.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[1919]:


X_validation.columns


# In[1920]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score


# In[1921]:


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 5, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

svc_param_selection(X_train,y_train, 5)


# {'C': 5, 'gamma': 0.001}

# In[1922]:


# optimal parameter combination for SVM model 
model = SVC(C=5, gamma=0.001, kernel='rbf')


# In[1923]:


model.fit(X_train,y_train)


# In[1924]:


predictions = model.predict(X_test)


# In[1925]:


print(confusion_matrix(y_test,predictions))


# In[1926]:


target_names = ['BUILDING_ID 0', 'BUILDING_ID 1', 'BUILDING_ID 2']

print(classification_report(y_test,predictions,target_names=target_names))


# In[1927]:


accuracy_score(y_test,predictions)


# In[1928]:


cohen_kappa_score(predictions,y_test)


# In[1929]:


# testing model with validation data set

predictions = model.predict(X_validation)


# In[1930]:


print(confusion_matrix(y_validation,predictions))


# In[1931]:


target_names = ['BUILDING_ID 0', 'BUILDING_ID 1', 'BUILDING_ID 2']

print(classification_report(y_validation,predictions,target_names=target_names))


# In[1932]:


accuracy_score(y_validation,predictions)


# In[1933]:


cohen_kappa_score(predictions, y_validation)


# ### PREDICTING BUILDING ID
# # KNN

# In[1934]:


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

def knn_param_selection(X, y, nfolds):
    K = [1,2,3,4,5]
    
    param_grid = {'n_neighbors': K}
    grid_search = GridSearchCV(knn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


knn_param_selection(X_train,y_train, 5)


# {'n_neighbors': 3}
# 

# In[1935]:


#optimal model
model = neighbors.KNeighborsClassifier(n_neighbors = 3)


# In[1936]:


import warnings
warnings.filterwarnings('ignore')
model.fit(X_train,y_train)


# In[1937]:


predictions = model.predict(X_test)


# In[1938]:


print(confusion_matrix(y_test,predictions))


# In[1939]:


target_names = ['BUILDING_ID 0', 'BUILDING_ID 1', 'BUILDING_ID 2']

print(classification_report(y_test,predictions,target_names=target_names))


# In[1940]:


accuracy_score(y_test,predictions)


# In[1941]:


cohen_kappa_score(predictions,y_test)


# In[1942]:


predictions = model.predict(X_validation)


# In[1943]:


print(confusion_matrix(y_validation,predictions))


# In[1944]:


target_names = ['BUILDING_ID 0', 'BUILDING_ID 1', 'BUILDING_ID 2']

print(classification_report(y_validation,predictions,target_names=target_names))


# In[1945]:


accuracy_score(y_validation,predictions)


# In[1946]:


cohen_kappa_score(predictions,y_validation)


# ### BUILDING ID 
# ### Random Forest

# In[1947]:


from sklearn.ensemble import RandomForestClassifier


# In[1948]:


RF = RandomForestClassifier()

def RF_param_selection(X, y, nfolds):
    trees = [50,55,60,65,70,75,80,85,90,95,100]
    
    param_grid = {'n_estimators': trees}
    grid_search = GridSearchCV(RF, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


RF_param_selection(X_train,y_train, 5)


# {'n_estimators': 50}

# In[1949]:


#optimal model
model = RandomForestClassifier(n_estimators=50)


# In[1950]:


model.fit(X_train,y_train)


# In[1951]:


predictions = model.predict(X_test)


# In[1952]:


print(confusion_matrix(y_test,predictions))


# In[1953]:


target_names = ['BUILDING_ID 0', 'BUILDING_ID 1', 'BUILDING_ID 2']

print(classification_report(y_test,predictions,target_names=target_names))


# In[1954]:


accuracy_score(y_test,predictions)


# In[1955]:


cohen_kappa_score(predictions,y_test)


# In[1956]:


predictions = model.predict(X_validation)


# In[1957]:


print(confusion_matrix(y_validation,predictions))


# In[1958]:


target_names = ['BUILDING_ID 0', 'BUILDING_ID 1', 'BUILDING_ID 2']

print(classification_report(y_validation,predictions,target_names=target_names))


# In[1959]:


accuracy_score(y_validation,predictions)


# In[1960]:


cohen_kappa_score(predictions,y_validation)


# ### Predicting 'FLOOR"

# In[1961]:


Wifi_trainc = Wifi_train
Wifi_validationc = Wifi_validation


# In[1962]:


#separate dependent(target) variable and independent variables

target = Wifi_trainc.loc[:, Wifi_train.columns == 'FLOOR']

df = Wifi_trainc.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)


# In[1963]:


#splitting validation data set
y_validation = Wifi_validationc.loc[:, Wifi_validationc.columns == 'FLOOR']


X_validation = Wifi_validationc.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[1964]:


knn = neighbors.KNeighborsClassifier()

def knn_param_selection(X, y, nfolds):
    K = [2,3,4,5]
    
    param_grid = {'n_neighbors': K}
    grid_search = GridSearchCV(knn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


knn_param_selection(X_train,y_train, 5)


# {'n_neighbors': 3}

# In[1965]:


#optimal model
model = neighbors.KNeighborsClassifier(n_neighbors = 3)


# In[1966]:


model.fit(X_train,y_train)


# In[1967]:


predictions = model.predict(X_test)


# In[1968]:


print(confusion_matrix(y_test,predictions))


# In[1969]:


target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3','FLOOR 4']

print(classification_report(y_test,predictions,target_names=target_names))


# In[1970]:


accuracy_score(y_test,predictions)


# In[1971]:


cohen_kappa_score(predictions,y_test)


# In[1972]:


predictions = model.predict(X_validation)


# In[1973]:


print(confusion_matrix(y_validation,predictions))


# In[1974]:


target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3','FLOOR 4']

print(classification_report(y_validation,predictions,target_names=target_names))


# In[1975]:


accuracy_score(y_validation,predictions)


# In[1976]:


cohen_kappa_score(predictions,y_validation)


# ### Predicting FLOOR
# 
# ### Random Forest

# In[1977]:


RF = RandomForestClassifier()

def RF_param_selection(X, y, nfolds):
    trees = [40,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140]
    
    param_grid = {'n_estimators': trees}
    grid_search = GridSearchCV(RF, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


RF_param_selection(X_train,y_train, 5)


# {'n_estimators': 65}

# In[1978]:


#optimal model
model = RandomForestClassifier(n_estimators=65)


# In[1979]:


model.fit(X_train,y_train)


# In[1980]:


predictions = model.predict(X_test)


# In[1981]:


print(confusion_matrix(y_test,predictions))


# In[1982]:


target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3','FLOOR 4']

print(classification_report(y_test,predictions,target_names=target_names))


# In[1983]:


accuracy_score(y_test,predictions)


# In[1984]:


cohen_kappa_score(predictions,y_test)


# In[1985]:


predictions = model.predict(X_validation)


# In[1986]:


print(confusion_matrix(y_validation,predictions))


# In[1987]:


target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3','FLOOR 4']

print(classification_report(y_validation,predictions,target_names=target_names))


# In[1988]:


accuracy_score(y_validation,predictions)


# In[1989]:


cohen_kappa_score(predictions,y_validation)


# ### Predicting FLOOR separately for every building

# In[1990]:


### Splitting Data Set for each Building to predict floor ###


# In[2005]:


Wifi_BuildingID_0 = Wifi_train[Wifi_train["BUILDINGID"]==0]
Wifi_BuildingID_1 = Wifi_train[Wifi_train["BUILDINGID"]==1]
Wifi_BuildingID_2 = Wifi_train[Wifi_train["BUILDINGID"]==2]

Wifi_v_BUILDING_0 = Wifi_validation[Wifi_validation["BUILDINGID"]==0]
Wifi_v_BUILDING_1 = Wifi_validation[Wifi_validation["BUILDINGID"]==1]
Wifi_v_BUILDING_2 = Wifi_validation[Wifi_validation["BUILDINGID"]==2]


# In[2007]:


Wifi_BuildingID_0['FLOOR'].value_counts()


# ####  Predictions of Floor for Building 0

# In[2064]:


#splitting Data Set

#separate dependent(target) variable and independent variables

target = Wifi_BuildingID_0.loc[:, Wifi_BuildingID_0.columns == 'FLOOR']

df = Wifi_BuildingID_0.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)

#splitting validation data set
y_validation = Wifi_v_BUILDING_0.loc[:, Wifi_validationc.columns == 'FLOOR']


X_validation = Wifi_v_BUILDING_0.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[2065]:


#optimal model
model = RandomForestClassifier(n_estimators=65)
model.fit(X_train,y_train)
predictionsB0 = model.predict(X_validation)


# In[2066]:


accuracy_score(y_validation,predictionsB0)


# In[2067]:


cohen_kappa_score(predictionsB0,y_validation)


# In[2068]:


#model evaluation

print(confusion_matrix(y_validation,predictionsB0))

target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3']


# In[2069]:


print(classification_report(y_validation,predictionsB0,target_names=target_names))


# #### Predictions of Floor for Building 01

# In[2052]:


#splitting Data Set

#separate dependent(target) variable and independent variables

target = Wifi_BuildingID_1.loc[:, Wifi_BuildingID_1.columns == 'FLOOR']

df = Wifi_BuildingID_1.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)

#splitting validation data set
y_validation = Wifi_v_BUILDING_1.loc[:, Wifi_validationc.columns == 'FLOOR']


X_validation = Wifi_v_BUILDING_1.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[2053]:


#optimal model
model = RandomForestClassifier(n_estimators=65)
model.fit(X_train,y_train)
predictionsB1 = model.predict(X_validation)


# In[2054]:


accuracy_score(y_validation,predictionsB1)


# In[2055]:


cohen_kappa_score(predictionsB1,y_validation)


# In[2056]:


print(confusion_matrix(y_validation,predictionsB1))


# In[2058]:


target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3']
print(classification_report(y_validation,predictionsB1,target_names=target_names))


# #### Predictions of Floor for Buildung 02

# In[2049]:


#splitting Data Set

#separate dependent(target) variable and independent variables

target = Wifi_BuildingID_2.loc[:, Wifi_BuildingID_2.columns == 'FLOOR']

df = Wifi_BuildingID_2.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)

#splitting validation data set
y_validation = Wifi_v_BUILDING_2.loc[:, Wifi_validationc.columns == 'FLOOR']


X_validation = Wifi_v_BUILDING_2.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[2025]:


#optimal model
model = RandomForestClassifier(n_estimators=65)
model.fit(X_train,y_train)
predictionsB2 = model.predict(X_validation)


# In[2026]:


accuracy_score(y_validation,predictionsB2)


# In[2027]:


cohen_kappa_score(predictionsB2,y_validation)


# In[2028]:


print(confusion_matrix(y_validation,predictionsB2))


# In[2029]:


target_names = ['FLOOR 0', 'FLOOR 1', 'FLOOR 2','FLOOR 3','FLOOR 4']
print(classification_report(y_validation,predictionsB2,target_names=target_names))


# ##### Predictions wrongly in Floor 3 instead of Floor 4

# In[2059]:


Wifi_v_BUILDING_2['FLOOR_pred'] = predictionsB2


# In[2060]:


Wifi_v_BUILDING_1['FLOOR_pred'] = predictionsB1


# In[2070]:


Wifi_v_BUILDING_0['FLOOR_pred'] = predictionsB0


# In[2071]:


wrong_pred_BU2 = Wifi_v_BUILDING_2[Wifi_v_BUILDING_2['FLOOR'] != Wifi_v_BUILDING_2['FLOOR_pred']]
wrong_pred_BU1 = Wifi_v_BUILDING_1[Wifi_v_BUILDING_1['FLOOR'] != Wifi_v_BUILDING_1['FLOOR_pred']]
wrong_pred_BU0 = Wifi_v_BUILDING_0[Wifi_v_BUILDING_0['FLOOR'] != Wifi_v_BUILDING_0['FLOOR_pred']]


# In[1858]:


#Wifi_plot = Wifi_train.copy()
raw_train_data['USERID'] = raw_train_data['USERID'].astype('int64')


# In[2072]:


#plotting wrong predictions for Floor (in red) and correct predictions (green)

fig, ax = plt.subplots(1, 1, figsize = (14, 5))

sns.scatterplot(Wifi_validation['LONGITUDE'],Wifi_validation['LATITUDE'],color='green')
sns.scatterplot(wrong_pred_BU0['LONGITUDE'],wrong_pred_BU0['LATITUDE'],color='red',alpha=0.5)
sns.scatterplot(wrong_pred_BU1['LONGITUDE'],wrong_pred_BU1['LATITUDE'],color='red',alpha=0.5)
sns.scatterplot(wrong_pred_BU2['LONGITUDE'],wrong_pred_BU2['LATITUDE'],color='red',alpha=0.5)


# In[1848]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2073]:


fig = plt.figure(figsize = (17, 4))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Wifi_validation['LATITUDE'],Wifi_validation['LONGITUDE'],Wifi_validation['FLOOR'],alpha=0.15,color='green')
ax.scatter(wrong_pred_BU0['LATITUDE'],wrong_pred_BU0['LONGITUDE'],wrong_pred_BU0['FLOOR'], c='red')
ax.scatter(wrong_pred_BU1['LATITUDE'],wrong_pred_BU1['LONGITUDE'],wrong_pred_BU1['FLOOR'], c='red')
ax.scatter(wrong_pred_BU2['LATITUDE'],wrong_pred_BU2['LONGITUDE'],wrong_pred_BU2['FLOOR'], c='red')

ax.view_init(25,5)
plt.show()


# ### Prediciting Longitude

# In[2086]:


Wifi_trainc = Wifi_train
Wifi_validationc = Wifi_validation


# In[2087]:


#separate dependent(target) variable and independent variables

target = Wifi_trainc.loc[:, Wifi_trainc.columns == 'LONGITUDE']

df = Wifi_trainc.drop(['BUILDINGID','FLOOR','LATITUDE','BUILDINGID','LONGITUDE'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)


# In[2088]:


#splitting validation data set
y_validation = Wifi_validationc.loc[:, Wifi_validationc.columns == 'LONGITUDE']


X_validation = Wifi_validationc.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[2089]:


X_validation.columns


# ### KNN

# In[2090]:


from sklearn.neighbors import KNeighborsRegressor


# In[2091]:


knn = KNeighborsRegressor()

def knn_param_selection(X, y, nfolds):
    K = [2,3,4,5]
    
    param_grid = {'n_neighbors': K}
    grid_search = GridSearchCV(knn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


knn_param_selection(X_train,y_train, 5)


# In[2092]:


#optimal model
model = KNeighborsRegressor(n_neighbors=2)


# In[2093]:


model.fit(X_train,y_train)


# In[2094]:


prediction_l_knn = model.predict(X_test)


# In[2095]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,prediction_l_knn)


# In[2096]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,prediction_l_knn)


# In[2097]:


from sklearn.metrics import r2_score

r2_score(y_test,prediction_l_knn)


# In[2098]:


prediction_l_knn = model.predict(X_validation)


# In[2099]:


mean_absolute_error(y_validation,prediction_l_knn)


# In[2100]:


mean_squared_error(y_validation,prediction_l_knn)


# In[2101]:


r2_score(y_validation, prediction_l_knn)


# ### Random Forest

# In[2102]:


from sklearn.ensemble import RandomForestRegressor


# In[2103]:


RF = RandomForestRegressor()

def RF_param_selection(X, y, nfolds):
    trees = [50,55,60,65,70,75,80,85,90,95,100]
    
    param_grid = {'n_estimators': trees}
    grid_search = GridSearchCV(RF, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


RF_param_selection(X_train,y_train, 5)


# {'n_estimators': 90}

# In[2104]:


#optimal model
model = RandomForestRegressor(n_estimators = 90)


# In[2105]:


model.fit(X_train,y_train)


# In[2106]:


prediction_l_rf = model.predict(X_test)


# In[2107]:


mean_absolute_error(y_test,prediction_l_rf)


# In[2108]:


mean_squared_error(y_test,prediction_l_rf)


# In[2109]:


r2_score(y_test,prediction_l_rf)


# In[2110]:


prediction_l_rf = model.predict(X_validation)


# In[2111]:


mean_absolute_error(y_validation,prediction_l_rf)


# In[2112]:


mean_squared_error(y_validation,prediction_l_rf)


# In[2113]:


r2_score(y_validation, prediction_l_rf)


# ### Predicting Latitude

# ### KNN

# In[2114]:


#separate dependent(target) variable and independent variables

target = Wifi_train.loc[:, Wifi_train.columns == 'LATITUDE']

df = Wifi_train.drop(['BUILDINGID','FLOOR','LONGITUDE','BUILDINGID','LATITUDE'],axis=1)

#split Data
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)


# In[2115]:


#splitting validation data set
y_validation = Wifi_validation.loc[:, Wifi_validation.columns == 'LATITUDE']


X_validation = Wifi_validation.drop(['BUILDINGID','LONGITUDE','LATITUDE','FLOOR','USERID','PHONEID','TIMESTAMP','RELATIVEPOSITION','SPACEID'],axis=1)


# In[2116]:


X_validation.columns


# In[2117]:


knn = KNeighborsRegressor()

def knn_param_selection(X, y, nfolds):
    K = [2,3,4,5]
    
    param_grid = {'n_neighbors': K}
    grid_search = GridSearchCV(knn, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


knn_param_selection(X_train,y_train, 5)


# In[2118]:


#optimal model
model = KNeighborsRegressor(n_neighbors=2)


# In[2119]:


model.fit(X_train,y_train)


# In[2120]:


prediction_t_knn = model.predict(X_test)


# In[2121]:


mean_absolute_error(y_test,prediction_t_knn)


# In[2122]:


mean_squared_error(y_test,prediction_t_knn)


# In[2123]:


r2_score(y_test,prediction_t_knn)


# In[2124]:


prediction_t_knn = model.predict(X_validation)


# In[2125]:


mean_absolute_error(y_validation,prediction_t_knn)


# In[2126]:


mean_squared_error(y_validation,prediction_t_knn)


# In[2127]:


r2_score(y_validation, prediction_t_knn)


# ### Random Forest

# In[2130]:


RF = RandomForestRegressor()

def RF_param_selection(X, y, nfolds):
    trees = [50,55,60,65,70,75,80,85,90,95,100]
    
    param_grid = {'n_estimators': trees}
    grid_search = GridSearchCV(RF, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[ ]:


RF_param_selection(X_train,y_train, 5)


# {'n_estimators': 55}

# In[2131]:


#optimal model
model = RandomForestRegressor(n_estimators = 55)


# In[2132]:


model.fit(X_train,y_train)


# In[2133]:


prediction_t_rf = model.predict(X_test)


# In[2134]:


mean_absolute_error(y_test,prediction_t_rf)


# In[2135]:


mean_squared_error(y_test,prediction_t_rf)


# In[2136]:


r2_score(y_test,prediction_t_rf)


# In[2137]:


prediction_t_rf = model.predict(X_validation)


# In[2138]:


mean_absolute_error(y_validation,prediction_t_rf)


# In[2139]:


mean_squared_error(y_validation,prediction_t_rf)


# In[2140]:


r2_score(y_validation, prediction_t_rf)


# #### Plotting Position of Predictions for Long/Lat and actual Location

# In[2141]:


type(prediction_t_rf)


# In[2142]:


prediction_l_knn2 = list(prediction_l_knn)
prediction_t_knn2 = list(prediction_t_knn)


# In[2143]:


longlat = pd.DataFrame()


# In[2144]:


prediction_l_knn.shape


# In[2145]:


longlat['LONGITUDE']= prediction_l_knn2


# In[2146]:


longlat['LATITUDE']=prediction_t_knn2


# In[2149]:


#plotting predictions for coordinates (green) and real coordinates(black)

fig, ax = plt.subplots(1, 1, figsize = (12, 5))

sns.scatterplot(Wifi_validation['LONGITUDE'],Wifi_validation['LATITUDE'],color='black',alpha=1)
sns.scatterplot(longlat['LONGITUDE'],longlat['LATITUDE'],color='green',alpha=0.3,legend='full')

