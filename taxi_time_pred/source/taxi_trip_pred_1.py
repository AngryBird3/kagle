
# coding: utf-8

# In[26]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression # Our ML model
from sklearn.preprocessing import LabelEncoder # Preprocess to get float
import numpy as np # Numpy
from geopy.distance import vincenty # To calculate distance
from ast import literal_eval # This is to convert string representation of array to actual array


# In[10]:


df = pd.read_csv('../input/train.csv')


# ### Analyzing data

# What are all columns/features we have?

# In[11]:


df.columns


# Missing feature/value?

# In[12]:


df.columns[df.isnull().any()]


# Count per feature/column

# In[13]:


df.count()


# Let's see few rows of data ..

# In[15]:


sum(df.MISSING_DATA)


# In[ ]:


df = df[df.MISSING_DATA == False]
df.count()


# In[ ]:


# I need to calculate distance between starting and ending location,
# so what I'm gonna do is take POLYLINE column, and split it into multiple
# chunks- so that I can load entire column into memory and split it

# Convert list representation to list
#df['POLYLINE'] = df['POLYLINE'].apply(literal_eval)

# Now challenge is to partition data into say k:
polyline_chunks_str = np.array_split(df['POLYLINE'], 1000) # k = 5
# def str_to_arr(p):
#     return literal_eval(p)
def v_str_to_arr(arr_p):
    v = np.array([])
    for i in range(len(arr_p)):
        try:
            np.append(literal_eval(arr_p[i]))
        except:
            print(arr_p[i])
            break
            exit(-1)
for i in range(len(polyline_chunks_str)):
    p = v_str_to_arr(polyline_chunks_str[0])
    np.append(polyline, p)

#print(polyline_chunks_str[0][0]) #printing first 10 rows of 1st chunk
# polyline_chunks = np.array([])
# for p in polyline_chunks_str:
#     np.append(polyline_chunks, np.array(literal_eval(p)))


# In[24]:


# We need to calculate distance and remove this POLYLINE 
# Vectorize is fun - https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.vectorize.html
def distance(polyline):
    return vincenty(polyline[0], polyline[-1])
v_dist = np.vectorize(distance)
# Let's see how much time it takes for 1 chunk
distance_1 = v_dist(polyline_chunks[0])

