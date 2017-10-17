
# coding: utf-8

# In[3]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression # Our ML model
from sklearn.preprocessing import LabelEncoder # Preprocess to get float
import numpy as np # Numpy
from geopy.distance import vincenty # To calculate distance
from ast import literal_eval # This is to convert string representation of array to actual array


# In[4]:


df = pd.read_csv('../input/train.csv')


# ### Analyzing data

# What are all columns/features we have?

# In[5]:


df.columns


# Missing feature/value?

# In[6]:


df.columns[df.isnull().any()]


# Count per feature/column

# In[7]:


df.count()


# Let's see few rows of data ..

# In[8]:


sum(df.MISSING_DATA)


# In[ ]:


df.drop(df[df.MISSING_DATA == True].index, inplace=True)
df.count()


# In[ ]:


# I need to calculate distance between starting and ending location,
# so what I'm gonna do is take POLYLINE column, and split it into multiple
# chunks- so that I can load entire column into memory and split it

# Convert list representation to list
df['POLYLINE'] = df['POLYLINE'].apply(literal_eval)


# In[189]:


# We need to calculate distance and remove this POLYLINE 
# Vectorize is fun - https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.vectorize.html
def distance(polyline):
    try:
        return vincenty(polyline[0], polyline[-1]).miles
    except Exception as e:
        return float('nan')
# Let's see how much time it takes for 1 chunk
#dist_1 = v_dist(polyline_chunks[0])


# In[190]:


# Calculate distance using above method
df['DISTANCE'] = df['POLYLINE'].apply(distance)


# In[1]:


print(df.DISTANCE.head())
# Drop distances with "NaN"
df.drop(df[df.DISTANCE == float('nan')].index, inplace=True)


# In[2]:


# Calculate label
def trip_time(polyline):
    return (len(polyline) - 1) * 15
label = df['POLYLINE'].apply(trip_time)


# In[ ]:


# TODO
# We'll (or we can) do fancy stuff like getting hour of day and classify them as 
# (peak hours, ok hours, easy hour/night time). 
# Get the day of the week
# Get the month of the year
# this all can be done with given timestamp


# In[186]:


# Drop polyline, missing data
train = df.drop(['POLYLINE', 'MISSING_DATA', 'ORIGIN_CALL','TAXI_ID', 'TIMESTAMP', 'TRIP_ID'], 1)
train.drop('TRIP_TIME', 1, inplace=True)
train.columns


# In[187]:


# Okay, time for training with LinearRegression
lr = LinearRegression()
lr = lr.fit(train, label)

