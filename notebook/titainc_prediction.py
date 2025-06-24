#!/usr/bin/env python
# coding: utf-8

# ### Titanic - Machine Learning from Disaster
# 

# ### 1. Introduction
# 
# This project analyzes passenger data from the Titanic disaster to predict survival outcomes using machine learning. We use data cleaning, feature engineering, and a logistic regression model to predict whether a passenger survived.
# 
# Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

# ### 2. Import Libraries

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set(style="whitegrid")


# ### 3. load the dataset

# In[46]:


df = pd.read_csv(r"C:\Users\Roshni\Downloads\titanic-survival-prediction\data\train.csv")
df.head()


# ### 4. Basic data exploration

# In[47]:


df.shape
df.info()
df.describe(include='all')
df.isnull().sum()


# ### 5. Data Visualization

# In[48]:


# Survival by Sex
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Sex")
plt.show()

# Survival by Passenger Class
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Age distribution
sns.histplot(df["Age"].dropna(), kde=True, bins=30)
plt.title("Age Distribution")
plt.show()



# ###  6. Data Cleaning

# In[49]:


# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Drop Ticket and Name for simplicity
df.drop(['Ticket', 'Name'], axis=1, inplace=True)


# ### 7. Feature Engineering

# In[50]:


# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# ### 8. Prepare Data for Modeling

# In[51]:


X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### 9. Train Logistic Regression Model
# 
# 

# In[52]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ### 10. Evaluate Model

# In[53]:


# Accuracy and metrics
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ###  11. Conclusion
# 
# - The logistic regression model achieved an accuracy of around **81%**.
# - Gender and passenger class had strong influence on survival chances.
# - I cleaned missing values, encoded categorical variables, and created a simple prediction model.
# - Future improvements: try Random Forests, feature selection, hyperparameter tuning.
# 

# In[ ]:




