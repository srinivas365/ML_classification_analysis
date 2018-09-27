
# coding: utf-8

# In[219]:


import pandas as pd
import numpy as np


# In[220]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score


# In[221]:


dataset=pd.read_csv('train.csv')


# In[222]:


replace_dep={'3+':'3'}
dataset['Dependents'].replace(replace_dep,inplace=True)
dataset['Dependents'].fillna('Unknown',inplace=True)
dataset['Self_Employed'].fillna('Unknown',inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['Credit_History'].fillna('Unknown',inplace=True)
dataset['Married'].fillna('Unknown',inplace=True)
dataset['Gender'].fillna('Unknown',inplace=True)


# In[223]:


dataset.drop('Loan_ID',axis=1,inplace=True)
cols=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
train=pd.get_dummies(dataset,columns=cols)


# In[224]:


x=train.drop('Loan_Status',axis=1)
y=train.Loan_Status
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[225]:


model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[226]:


cm=confusion_matrix(y_test,y_pred)
ass=accuracy_score(y_pred,y_test)
print(cm,ass)


# In[227]:


datatest=pd.read_csv('test.csv')


# In[228]:


replace_dep={'3+':'3'}
datatest['Dependents'].replace(replace_dep,inplace=True)
datatest['Dependents'].fillna('Unknown',inplace=True)
datatest['Self_Employed'].fillna('Unknown',inplace=True)
datatest['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
datatest['Loan_Amount_Term'].fillna(dataset['LoanAmount'].mean(),inplace=True)
datatest['Credit_History'].fillna('Unknown',inplace=True)
datatest['Married'].fillna('Unknown',inplace=True)
datatest['Gender'].fillna('Unknown',inplace=True)


# In[229]:


submit=datatest['Loan_ID']
datatest.drop('Loan_ID',axis=1,inplace=True)
cols=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
test_set=pd.get_dummies(datatest,columns=cols)


# In[230]:


test_set['Married_Unknown']=0


# In[231]:


test_pred=model.predict(test_set)


# In[232]:


preds = pd.Series(test_pred)
submit = pd.concat([submit, preds], names=['Loan_ID', 'Loan_Status'], axis=1)
submit.columns = ['Loan_ID', 'Loan_Status']


# In[233]:


submit.to_csv('loan1.csv',index=False)

