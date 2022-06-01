#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
rng = np.random.default_rng(2022)

## Here's the information needed to do the first few tasks,
# which will give you some practice with basic Python methods

list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages  = [23, 24, 19, 87]
list_of_heights_cm = [175, 162, 178, 183]

for name in list_of_names:
  print("The name {:} is {:} letters long".format(name, len(name)))


########################################
# Here's the information for the second part, involving the linear
# classifier

# import the iris dataset as a pandas dataframe
iris_db = load_iris(as_frame=True) 
x_data = iris_db['data'] 
y_labels = iris_db['target'] # correct numeric labels
target_names = iris_db['target_names'] # string names

# Here's a starter example of plotting the data
fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
l_colors = ['maroon', 'darkgreen', 'blue']
for n, species in enumerate(target_names):    
  plt.scatter(x_data[y_labels==n].iloc[:,0], 
              x_data[y_labels==n].iloc[:,1], 
              c=l_colors[n], label=target_names[n])
plt.xlabel(iris_db['feature_names'][0])
plt.ylabel(iris_db['feature_names'][1])
plt.grid(True)
plt.legend() # uses the 'label' argument passed to scatter()
plt.tight_layout()
# uncomment this line to show the figure, or use
# interactive mode -- plt.ion() --  in iPython
# plt.show()
plt.savefig('iris_data.png')


## A trivial example classifier.  You'll copy and modify this to 
# perform a linear classification function.
def classify_rand(x):    
  return rng.integers(0,2, endpoint=True)


# A function to measure the accuracy of a classifier and
# create a confusion matrix.  Keras and Scikit-learn have more sophisticated
# functions that do this, but this simple version will work for
# this assignment.
def evaluate_classifier(cls_func, x_data, labels, print_confusion_matrix=True):
  n_correct = 0
  n_total = x_data.shape[0]
  cm = np.zeros((3,3))
  for i in range(n_total):
    x = x_data[i,:]
    y = cls_func(x)
    y_true = labels[i]
    cm[y_true, y] += 1
    if y == y_true:
      n_correct += 1    
    acc = n_correct / n_total
  print(f"Accuracy = {n_correct} correct / {n_total} total = {100.0*acc:3.2f}%")
  if print_confusion_matrix:
    print(f"{12*' '}Estimated Labels")
    print(f"              {0:3.0f}  {1.0:3.0f}  {2.0:3.0f}")
    print(f"{12*' '} {15*'-'}")
    print(f"True    0 |   {cm[0,0]:3.0f}  {cm[0,1]:3.0f}  {cm[0,2]:3.0f} ")
    print(f"Labels: 1 |   {cm[1,0]:3.0f}  {cm[1,1]:3.0f}  {cm[1,2]:3.0f} ")
    print(f"        2 |   {cm[2,0]:3.0f}  {cm[2,1]:3.0f}  {cm[2,2]:3.0f} ")
    print(f"{40*'-'}")
  ## done printing confusion matrix  

  return acc, cm


# In[55]:


# Creating list of lenghts using list comprehension
list_of_lengths  = [ len(name) for name in list_of_names]
print(list_of_lengths)


# In[56]:


# Importing class "person"
from person import person


# In[57]:


# Making people dictionary with names as key, and person object as value
people = {}
for (name, age, height) in zip(list_of_names, list_of_ages, list_of_heights_cm):
    people[name] = person (name = name, age_yrs = age, height = height)
    


# In[58]:


# Just testing the dictionary
people["Luisa"].__repr__()


# In[59]:


# Converting ages list to an numpy array
ages_array = np.array(list_of_ages)
print(ages_array)


# In[60]:


# Converting heights list to an numpy array
heights_array = np.array(list_of_heights_cm)
print(heights_array)


# In[61]:


# Finding the average of the ages
average_age = np.mean(ages_array)
print(average_age)


# In[62]:


# Plotting a scatter plot of heights vs ages 
fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
plt.scatter(ages_array, heights_array)
plt.grid()
plt.title("Height vs Age")
plt.xlabel("Age")
plt.ylabel("Height")
plt.savefig("scatter_plot.png")


# In[63]:


# Exploring the Iris dataset
# import the iris dataset as a pandas dataframe
iris_db = load_iris(as_frame=True) 
x_data = iris_db['data'] 
y_labels = iris_db['target'] # correct numeric labels
target_names = iris_db['target_names'] # string names

# Here's a starter example of plotting the data
fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
l_colors = ['maroon', 'darkgreen', 'blue']
for n, species in enumerate(target_names):    
  plt.scatter(x_data[y_labels==n].iloc[:,2], 
              x_data[y_labels==n].iloc[:,3], 
              c=l_colors[n], label=target_names[n])
plt.xlabel(iris_db['feature_names'][2])
plt.ylabel(iris_db['feature_names'][3])
plt.grid(True)
plt.legend() # uses the 'label' argument passed to scatter()
plt.tight_layout()
# uncomment this line to show the figure, or use
# interactive mode -- plt.ion() --  in iPython
# plt.show()
plt.savefig('iris_data2.png')


# In[138]:


# Defining the Linear classifier which simply calculate argmax(weight*feature + bias)
def classify_iris(features): 
    w = np.array([[0.2, 0.1, 1.8, -0.3], [-0.1, 0.07, 0.3, 1.2], [0.09, -0.08, 2.1, 0.3]])    
    b = np.array([0.9, -0.5, 0.8])
    return np.argmax (w@features.T + b)


# In[139]:


# Just testing with a random input to make sure the function works correctly
classify_iris(np.array([1.0, 2.0, 3.0, 4.0]))


# In[140]:


# Converting data to a numpy array
x_data_arr = np.array(x_data)


# In[141]:


# Using the evalute function that is provided. You can see that the accuracy is 58%
evaluate_classifier(classify_iris, x_data_arr, y_labels, print_confusion_matrix=True)

## Now evaluate the classifier we've built.  This will evaluate the
# random classifier, which should have accuracy around 33%.
acc, cm = evaluate_classifier(classify_iris, x_data.to_numpy(), y_labels.to_numpy())