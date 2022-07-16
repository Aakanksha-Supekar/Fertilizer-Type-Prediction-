#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")


# # Reading Dataset

# In[2]:


df = pd.read_csv("/home/aakanksha/Downloads/Fertilizer Prediction.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df['Soil Type'].unique()


# # visualizing data
# 

# In[6]:


import seaborn as sns
sns.countplot(x='Soil Type', data = df)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
sns.countplot(x='Crop Type', data = df)


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
sns.countplot(x='Fertilizer Name', data = df)


# In[9]:


#Defining function for Continuous and catogorical variable
def plot_conti(x):
    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(15,5),tight_layout=True)
    axes[0].set_title('Histogram')
    sns.histplot(x,ax=axes[0])
    axes[1].set_title('Checking Outliers')
    sns.boxplot(x,ax=axes[1])
    axes[2].set_title('Relation with output variable')
    sns.boxplot(y = x,x = df['Fertilizer Name'])
    
def plot_cato(x):
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5),tight_layout=True)
    axes[0].set_title('Count Plot')
    sns.countplot(x,ax=axes[0])
    axes[1].set_title('Relation with output variable')
    sns.countplot(x = x,hue = df['Fertilizer Name'], ax=axes[1])


# In[10]:


#EDA - Temparature variable
plot_conti(df['Temparature'])


# In[11]:


#EDA - Humidity variable
plot_conti(df['Humidity '])


# In[12]:


#EDA - Moisture variable
plot_conti(df['Moisture'])


# In[13]:


plot_cato(df['Soil Type'])


# In[14]:


#relation of soil type with Temperature 
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Soil Type'],y=df['Temparature'])


# In[15]:


#relation of soil type and Temperature with output variable
plt.figure(figsize=(15,6))
sns.boxplot(x=df['Soil Type'],y=df['Temparature'],hue=df['Fertilizer Name'])


# In[16]:


#EDA - Crop_Type variable
plot_cato(df['Crop Type'])


# In[17]:


#relation of crop type with temperature
plt.figure(figsize=(15,6))
sns.boxplot(x=df['Crop Type'],y=df['Temparature'])


# In[18]:


#relation of crop type with Humidity
plt.figure(figsize=(15,8))
sns.boxplot(x=df['Crop Type'],y=df['Humidity '])


# In[19]:


#EDA - Nitrogen variable
plot_conti(df['Nitrogen'])


# In[20]:


#relation of nitrogen wrt to crop type
plt.figure(figsize=(15,8))
sns.boxplot(x=df['Crop Type'],y=df['Nitrogen'])


# In[21]:


#EDA - Potassium variable
plot_conti(df['Potassium'])


# In[22]:


#EDA - Phosphorous variable
plot_conti(df['Phosphorous'])


# # Preprocessing of data

# In[23]:


y = df['Fertilizer Name'].copy()
X = df.drop('Fertilizer Name', axis=1).copy()


# In[24]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3,4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[25]:


X[0]


# # Train - test - split

# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)


# # Feature Scaling

# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[28]:


X_train[0]


# # Random Forest Classifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 100, criterion = 'gini' , random_state= 42)
classifier.fit(X_train, y_train)


# In[30]:


y_pred = classifier.predict(X_test)


# # Creating confusion matrix

# In[31]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[32]:


classifier.score(X_test, y_test)


# Test accuracy = 96.67%

# # Preprocessing using Label Encoder

# In[33]:


#encoding the labels for categorical variables
from sklearn.preprocessing import LabelEncoder


# In[34]:


#encoding Soil Type variable
encode_soil = LabelEncoder()
df['Soil Type'] = encode_soil.fit_transform(df['Soil Type'])

#creating the DataFrame
Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
Soil_Type = Soil_Type.set_index('Original')
Soil_Type


# In[35]:


encode_crop =  LabelEncoder()
df['Crop Type'] = encode_crop.fit_transform(df['Crop Type'])

#creating the DataFrame
Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
Crop_Type = Crop_Type.set_index('Original')
Crop_Type


# In[36]:


encode_ferti = LabelEncoder()
df['Fertilizer Name'] = encode_ferti.fit_transform(df['Fertilizer Name'])

#creating the DataFrame
Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
Fertilizer = Fertilizer.set_index('Original')
Fertilizer


# In[37]:


#splitting the data into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop('Fertilizer Name',axis=1),df['Fertilizer Name'],test_size=0.2,random_state=1)
print('Shape of Splitting :')
print('x_train = {}, y_train = {}, x_test = {}, y_test = {}'.format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))


# In[38]:


x_train.info()


# # Random Forest Classifier

# In[39]:


rand = RandomForestClassifier(random_state = 42)
rand.fit(x_train,y_train)


# In[40]:


pred_rand = rand.predict(x_test)


# # Hyperparameter tuning with GridSearchCV

# In[41]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

params = {
    'n_estimators':[300,400,500],
    'max_depth':[5,10,15],
    'min_samples_split':[2,5,8]
}
grid_rand = GridSearchCV(rand,params,cv=3,verbose=3,n_jobs=-1)

grid_rand.fit(x_train,y_train)

pred_rand = grid_rand.predict(x_test)

print(classification_report(y_test,pred_rand))

print('Best score : ',grid_rand.best_score_)
print('Best params : ',grid_rand.best_params_)


# Best score = 97.48%

# In[42]:


y_train[2]


# In[43]:


#pickling the file
import pickle
pickle_out = open('classifier.pkl','wb')
pickle.dump(grid_rand,pickle_out)
pickle_out.close()


# In[44]:


#pickling the file
import pickle
pickle_out = open('fertilizer.pkl','wb')
pickle.dump(encode_ferti,pickle_out)
pickle_out.close()


# In[45]:


df.head()


# In[46]:


model = pickle.load(open('classifier.pkl','rb'))
ferti = pickle.load(open('fertilizer.pkl','rb'))
ferti.classes_[6]

ans = model.predict([[32 ,	62, 	34 ,	3 ,	9 ,	22 ,	0 ,	20]])
if ans[0] == 0:
    print("10-26-26")
elif ans[0] ==1:
    print("14-35-14")
elif ans[0] == 2:
    print("17-17-17	")
elif ans[0] == 3:
    print("20-20")
elif ans[0] == 4:
    print("28-28")
elif ans[0] == 5:
    print("DAP")
else:
    print("Urea")


# In[ ]:




