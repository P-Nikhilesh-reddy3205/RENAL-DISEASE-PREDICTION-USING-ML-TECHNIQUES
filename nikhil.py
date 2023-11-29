#!/usr/bin/env python
# coding: utf-8

# # DATA ACCUMULATION

# In[1]:


#Data Accumulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 26)


# In[2]:


df= pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/B1/CKD_Data.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.drop('id', axis = 1, inplace = True)


# In[5]:


df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# # DATA PRE-PROCESSING

# In[9]:


#DATA PRE PROCESSING
# converting necessary columns to numerical type

df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')


# In[10]:


df.info()


# In[11]:


# Extracting categorical and numerical columns

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']
# looking at unique values in categorical columns

for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")


# In[12]:



# replace incorrect values

df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)

df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')

df['class'] = df['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
df['class'] = pd.to_numeric(df['class'], errors='coerce')
cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']

for col in cols:
    print(f"{col} has {df[col].unique()} values\n")


# # FEATURE SELECTION

# In[13]:


# FEATURE SELECTION
# checking numerical features distribution

plt.figure(figsize = (20, 15))
plotnumber = 1

for column in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[14]:




# looking at categorical columns

plt.figure(figsize = (20, 15))
plotnumber = 1

for column in cat_cols:
    if plotnumber <= 11:
        ax = plt.subplot(3, 4, plotnumber)
        sns.countplot(df[column], palette = 'rocket')
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[15]:


# heatmap of data

plt.figure(figsize = (15, 8))

sns.heatmap(df.corr(), annot = True, linewidths = 2, linecolor = 'lightgrey')
plt.show()


# In[16]:


df.columns


# # CREATING PLOTS

# In[17]:


# defining functions to create plot

def violin(col):
    fig = px.violin(df, y=col, x="class", color="class", box=True, template = 'plotly_dark')
    return fig.show()

def kde(col):
    grid = sns.FacetGrid(df, hue="class", height = 6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    
def scatter(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color="class", template = 'plotly_dark')
    return fig.show()
violin('red_blood_cell_count')


# In[18]:


kde('red_blood_cell_count')


# In[19]:


violin('white_blood_cell_count')


# In[20]:


kde('white_blood_cell_count')


# In[21]:


violin('packed_cell_volume')


# In[22]:


kde('packed_cell_volume')


# In[23]:


violin('haemoglobin')


# In[24]:


kde('haemoglobin')


# In[25]:


violin('albumin')


# In[26]:


kde('albumin')


# In[27]:


violin('blood_glucose_random')


# In[28]:


kde('blood_glucose_random')


# In[29]:


violin('sodium')


# In[30]:


kde('sodium')


# In[31]:


violin('blood_urea')


# In[32]:


kde('blood_urea')


# In[33]:


violin('specific_gravity')


# In[34]:


kde('specific_gravity')


# In[35]:


scatter('haemoglobin', 'packed_cell_volume')


# In[36]:


scatter('red_blood_cell_count', 'packed_cell_volume')


# In[37]:


scatter('red_blood_cell_count', 'albumin')


# In[38]:


scatter('sugar', 'blood_glucose_random')


# In[39]:


scatter('packed_cell_volume','blood_urea')


# In[40]:


px.bar(df, x="specific_gravity", y="packed_cell_volume", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[41]:


px.bar(df, x="specific_gravity", y="albumin", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[42]:


px.bar(df, x="blood_pressure", y="packed_cell_volume", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[43]:


px.bar(df, x="blood_pressure", y="haemoglobin", color='class', barmode='group', template = 'plotly_dark', height = 400)


# In[44]:


# checking for null values

df.isna().sum().sort_values(ascending = False)


# In[45]:


df[num_cols].isnull().sum()


# In[46]:


df[cat_cols].isnull().sum()


# In[47]:


# filling null values, we will use two methods, random sampling for higher null values and 
# mean/mode sampling for lower null values

def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample
    
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)


# In[48]:


# filling num_cols null values using random sampling method

for col in num_cols:
    random_value_imputation(col)
df[num_cols].isnull().sum()


# In[49]:


# filling "red_blood_cells" and "pus_cell" using random sampling method and rest of cat_cols using mode imputation

random_value_imputation('red_blood_cells')
random_value_imputation('pus_cell')

for col in cat_cols:
    impute_mode(col)
df[cat_cols].isnull().sum()


# In[50]:


for col in cat_cols:
    print(f"{col} has {df[col].nunique()} categories\n")


# In[51]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
df.head()


# # MODEL DEVELOPMENT

# In[52]:


# MODEL DEVELOPMENT
ind_col = [col for col in df.columns if col != 'class']
dep_col = 'class'

X = df[ind_col]
y = df[dep_col]
# splitting data intp training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# # MODEL EVALUATIONS

# In[53]:


# MODEL EVALUATIONS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of knn

knn_acc = accuracy_score(y_test, knn.predict(X_test))

print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
print(f"Test Accuracy of KNN is {knn_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")


# In[54]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


# In[55]:


# hyper parameter tuning of decision tree 

from sklearn.model_selection import GridSearchCV
grid_param = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'splitter' : ['best', 'random'],
    'min_samples_leaf' : [1, 2, 3, 5, 7],
    'min_samples_split' : [1, 2, 3, 5, 7],
    'max_features' : ['auto', 'sqrt', 'log2']
}

grid_search_dtc = GridSearchCV(dtc, grid_param, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dtc.fit(X_train, y_train)


# In[56]:


# best parameters and best score

print(grid_search_dtc.best_params_)
print(grid_search_dtc.best_score_)


# In[57]:


# best estimator

dtc = grid_search_dtc.best_estimator_

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")


# In[58]:


from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rd_clf.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of random forest

rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")


# In[59]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of ada boost

ada_acc = accuracy_score(y_test, ada.predict(X_test))

print(f"Training Accuracy of Ada Boost Classifier is {accuracy_score(y_train, ada.predict(X_train))}")
print(f"Test Accuracy of Ada Boost Classifier is {ada_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, ada.predict(X_test))}")


# In[60]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of gradient boosting classifier

gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")


# In[61]:


sgb = GradientBoostingClassifier(max_depth = 4, subsample = 0.90, max_features = 0.75, n_estimators = 200)
sgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier

sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")


# In[ ]:





# In[62]:


#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
a = LogisticRegression()
a.fit(X_train, y_train)
a_acc = accuracy_score(y_test, a.predict(X_test))
print(f"Training Accuracy of Logistic Regression is {accuracy_score(y_train, a.predict(X_train))}")
print(f"Test Accuracy of Logistic Regression is {a_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, a.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, a.predict(X_test))}")


# In[ ]:





# In[63]:


from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")


# In[64]:


pred = True


# # PERFORMANCE METRICS

# In[65]:


# PERFORMANCE METRICS
models = pd.DataFrame({
    'Model' : [ 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','Ada Boost Classifier',
             'Gradient Boosting Classifier', 'Stochastic Gradient Boosting',  'Extra Trees Classifier', 'Logistic Regression'],
    'Score' : [knn_acc, dtc_acc, rd_clf_acc, ada_acc, gb_acc, sgb_acc, etc_acc,a_acc]
})


models.sort_values(by = 'Score', ascending = False)


# # MODELS COMPARISON

# In[66]:


px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Models Comparison')


# In[68]:


medication = pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/B1/medication.csv")
medication.head()


# # MODEL PREDICTIONS

# In[69]:


# MODEL PREDICTIONS
pred=dtc.predict(X_test)
pred=[1 if y>=0.5 else 0 for y in pred]

print("original  : {0}".format(", ".join(str(x)for x in y_test)))
print("predicted : {0}".format(", ".join(str(x) for x in pred)))


# In[70]:


pred=ada.predict(X_test)
pred=[1 if y>=0.5 else 0 for y in pred]

print("original  : {0}".format(", ".join(str(x)for x in y_test)))
print("predicted : {0}".format(", ".join(str(x) for x in pred)))


# # RESULTS

# # IF RESULT == POSITIVE

# In[71]:


# 1 means Renal Disease
# 0 means No Disease
if pred==1:
    medication = pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/medication.csv")
    medication.head()
print(medication.head())


# IF RESULT == NEGATIVE

# In[72]:


if pred == 0:
        medication = pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/medication.csv")
        medication.head()
print("STAY HEALTHY!")


# In[75]:


whiteBlood = (input("White Blood Cells:  "))
bloodUrea = (input("Blood Urea:  "))
bloodGlucose = (input("Blood Glucose:  "))
creatinine = (input("Serum Creatinine:  "))
cellVolume = (input("Packed Cell Volume:  "))
albumin = (input("Albumin:  "))
hemoglobin = (input("Hemoglobin:  "))
age= (input("Age:  "))
sugar= (input("Sugar:  "))
hypertension= (input("Hypertension:  "))
print(prediction())
if prediction==1:
    print("Result is Positive")
    print("Recommendations:")
    medication = pd.read_csv("C:/Users/nikhi/OneDrive/Desktop/medication.csv")
    medication.head()
print(medication.head())
print("Life Style changes:")
print("hai")


# In[74]:


def prediction():
    
    if whiteBlood and bloodUrea and bloodGlucose and cellVolume and albumin and hemoglobin and age and sugar and hypertension and creatinine:
        
        return 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




