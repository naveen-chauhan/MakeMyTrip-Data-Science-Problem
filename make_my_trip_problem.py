
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


train=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\make_my_trip\dataset\train.csv')
train.head()


# In[3]:


test=pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\make_my_trip\dataset\test.csv')


# In[4]:


test.head()


# In[5]:


ax=plt.subplots(figsize=(15,9))
correlation=train.corr()
sns.heatmap(correlation,vmax=0.8,square=True)


# In[6]:


output_value=train.P


# In[7]:


train.isnull().sum()


# In[8]:


train.shape


# In[9]:


#now do in binary
train.A.value_counts()


# In[10]:


#drop rows with less number of value
train.D.value_counts()


# In[11]:


train.E.value_counts()


# In[12]:


train.F.value_counts()


# In[13]:


train.G.value_counts()


# In[14]:


train.I.value_counts()


# In[15]:


train.J.value_counts()


# In[16]:


train.L.value_counts()


# In[17]:


train.M.value_counts()


# In[18]:


train.loc[train.D=='l',]


# In[19]:


train.O.hist(bins=50)


# In[20]:


#delete that rows 
len(train.loc[train.O>10000,])


# In[21]:


data=train.append(test)


# In[22]:


data.shape


# In[23]:


data.D.value_counts()


# In[24]:


data.E.value_counts()


# In[25]:


data.D=data.D.map({'u':0,'y':1,'l':2})
data.E=data.E.map({'g':0,'p':1,'gg':2})


# In[26]:


data.D.value_counts()


# In[27]:


ax=plt.subplots(figsize=(15,9))
correl=data.corr()
sns.heatmap(correl,vmax=.8,square=True)


# In[28]:


#now drop the column D or E
data.drop('D',inplace=True,axis=1)


# In[29]:


data.shape


# In[30]:


data.A=data.A.map({'a':1,'b':0})


# In[31]:


data.I=data.I.map({'f':1,'t':0})
data.J=data.J.map({'f':1,'t':0})
data.L=data.L.map({'f':1,'t':0})


# In[32]:


ax=plt.subplots(figsize=(15,9))
correl=data.corr()
sns.heatmap(correl,vmax=1,square=True)


# In[33]:


data.P.value_counts()


# In[34]:


data.I.value_counts()


# In[35]:


plt.subplot(1,1,1)
x=data['I']
y=data['P']
plt.plot(x,y,'o')
plt.plot(np.unique(x),np.poly1d(np.polyfit(x,y,1))(np.unique(x)))
plt.title('relationship between ')
plt.xlabel('I')
plt.ylabel('P')


# In[36]:


data.head()


# In[37]:


data.isnull().sum()


# In[38]:


train.isnull().sum()


# In[39]:


data.A.value_counts()


# In[40]:


data.A.fillna(1.0,inplace=True)


# In[41]:


data.B.hist(bins=50)


# In[42]:


data.H.hist(bins=50)


# In[43]:


data.H.loc[data.H>17,]


# In[44]:


len(data.B.loc[data.B>70,])


# In[45]:


B_mean=data.pivot_table(values='B', index='H')


# In[46]:


mean_value=data.groupby("H").B.mean()


# In[47]:


miss_bool = data['B'].isnull() 
data.loc[miss_bool,'B'] = data.loc[miss_bool,'H'].apply(lambda x: mean_value[x])


# In[48]:


data.isnull().sum()


# In[49]:


data.tail()


# In[50]:


data.F.value_counts()


# In[51]:


data.G.value_counts()


# In[52]:


data.drop('G',inplace=True,axis=1)


# In[53]:


data.F=data.F.map({'c':14,'q':1,'w':2,'i':3,'aa':4,'ff':5,'k':6,'cc':7,'x':8,'m':9,'d':10,'e':11,'j':12,'r':13})


# In[54]:


data.F.hist(bins=50)


# In[55]:


data.F.isnull().sum()


# In[56]:


data.head()


# In[57]:


data.M.value_counts()


# In[58]:


data.M=data.M.map({'g':0,'s':1,'p':2})


# In[59]:


data.isnull().sum()


# In[60]:


print(data.E.value_counts())
print(data.F.value_counts())


# In[61]:


data.E.fillna(0,inplace=True)


# In[62]:


data.F.values


# In[63]:


ax=plt.subplots(figsize=(15,9))
correl=data.corr()
sns.heatmap(correl,vmax=1,square=True)


# In[64]:


data.isnull().sum()


# In[65]:


new_data=data.loc[:,['H','I','J','K','P']]


# In[66]:


new_data.head()


# In[67]:


new_data.shape


# In[68]:


new_data.isnull().sum()


# In[69]:


#now split in train and test set
new_train=new_data.iloc[0:552,:]


# In[70]:


new_train.tail()


# In[71]:


new_test=new_data.iloc[552:,]


# In[72]:


new_test.drop('P',inplace=True,axis=1)


# In[73]:


new_test.head()


# In[74]:


target=new_train.P


# In[75]:


new_train.drop('P',inplace=True,axis=1)


# In[76]:


new_train.head()


# In[77]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(new_train,target,test_size=.20,random_state=42)


# In[82]:


train_X.head()


# In[83]:


test_X.head()


# In[86]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[90]:


model=[]
model.append(('log_reg',LogisticRegression()))
model.append(('svm',SVC()))
model.append(('rdc',RandomForestClassifier()))
model.append(('knn',KNeighborsClassifier()))
model.append(('dtc',DecisionTreeClassifier()))


# In[91]:


for name, clf in model:
    clf.fit(train_X,train_y)
    pred=clf.predict(test_X)
    print('accuracy of ',name," ",accuracy_score(test_y,pred))

