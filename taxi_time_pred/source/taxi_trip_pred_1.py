#!/usr/local/bin/python3
'''
https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii
'''
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression # Our ML model
from sklearn.preprocessing import LabelEncoder # Preprocess to get float
import numpy as np # Numpy
from geopy.distance import vincenty # To calculate distance
from ast import literal_eval # This is to convert string representation of array to actual array


def analysis(df): 
    # ### Analyzing data
    # What are all columns/features we have?
    print(df.columns)
    # Missing feature/value?
    print(df.columns[df.isnull().any()])
    # Count per feature/column
    print(df.count())
    # Let's see few rows of data ..
    print(sum(df.MISSING_DATA))

def preprocess(df):
    print("Removing missin data rows ..")
    df.drop(df[df.MISSING_DATA == True].index, inplace=True)
    print(df.count())
    # I need to calculate distance between starting and ending location,
    # so what I'm gonna do is take POLYLINE column, and split it into multiple
    # chunks- so that I can load entire column into memory and split it

    print("Convert list representation to list..")
    df['POLYLINE'] = df['POLYLINE'].apply(literal_eval)
    # We need to calculate distance and remove this POLYLINE 
    # Vectorize is fun - https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.vectorize.html
    def distance(polyline):
        try:
            return vincenty(polyline[0], polyline[-1]).miles
        except Exception as e:
            return float('nan')
    # Calculate distance using above method
    print("Calculating distance ..")
    df['DISTANCE'] = df['POLYLINE'].apply(distance)
    # Drop distances with "NaN"
    print("Dropping rows where we couldn't calculate distance - insufficient data?")
    df.drop(df[df.DISTANCE == float('nan')].index, inplace=True)

    # Calculate label
    print("Calculating label..")
    def trip_time(polyline):
        return (len(polyline) - 1) * 15
    label = df['POLYLINE'].apply(trip_time)

    # Add more feature like day of the week, hour of the day and month of the year.
    print("Adding more features - day_of_week, hour_of_day, month_of_year ..")
    df['my_dates'] = pd.to_datetime(df['TIMESTAMP'])
    df['day_of_week'] = df['my_dates'].dt.dayofweek
    df['month_of_year'] = df['my_dates'].dt.month
    df['hour_of_day'] = df['my_dates'].dt.hour

    # Drop polyline, missing data
    print("Dropping unnecessary features/columns ..")
    preprocess_df = df.drop(['POLYLINE', 'MISSING_DATA', 'ORIGIN_CALL','TAXI_ID', 'TIMESTAMP', 'TRIP_ID', 'my_dates'], 1)

    print(preprocess_df.columns)
    #LabelEncoder
    print("Label encoder ..")
    preprocess_df = preprocess_df.apply(LabelEncoder().fit_transform)
    return preprocess_df, label

def train_model(train, label):
    # Okay, time for training with LinearRegression
    lr = LinearRegression()
    lr = lr.fit(train, label)
    return lr

def predict(lr, test):
    predictions = lr.predict(test)
    return predictions

def taxi_time_pred(): 
    # Test df
    print("Reading training data frame..")
    train_df = pd.read_csv('../input/train.csv')
    print("Reading test data frame..")
    test_df = pd.read_csv('../input/test.csv')

    # Save TRIP_ID
    result_math_df = test_df[['TRIP_ID']].copy()

    train, train_label = preprocess(train_df)
    test, test_label = preprocess(test_df)
    print("\nAlmost done!!\n")
    # Using Math our labels...
    print("Spitting math label into result_math_df csv ..")
    result_math_df['TRAVEL_TIME'] = test_label
    result_math_df.to_csv('/Users/dhaaraab/Documents/spock/kagle/taxi_time_pred/source/predictions/result_math_df.csv', index=False)

    # Using ML model
    lr = train_model(train, train_label) 
    predictions = predict(lr, test)
    print("Spitting predicted label into result_regression csv")
    result_regression = result_math_df[['TRIP_ID']].copy()
    result_regression['TRAVEL_TIME'] = pd.Series(predictions, index=result_regression.index)
    result_regression.to_csv('/Users/dhaaraab/Documents/spock/kagle/taxi_time_pred/source/predictions/result_regression.csv', index=False)

taxi_time_pred()
