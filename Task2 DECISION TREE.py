#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[34]:


iris = pd.read_csv("Iris (1).csv")


# In[35]:


iris.head()


# In[36]:


iris.tail()


# In[37]:


iris.shape


# In[38]:


iris.describe()


# In[39]:


iris.info()


# In[40]:


iris.columns


# In[41]:


iris.isnull().sum()


# In[42]:


# Print the column names to verify the presence of 'Id'
print(iris.columns)

# Remove the 'Id' column if it exists
if 'Id' in iris.columns:
    iris.drop('Id', axis=1, inplace=True)


# In[43]:


# Split the dataset into features and target variable
X = iris.drop('Species', axis=1)
y = iris['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# Create an instance of the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)


# In[45]:


# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()


# In[ ]:




