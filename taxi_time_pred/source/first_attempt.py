#!/usr/local/bin/python3
'''
Problem: https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii
Resources:
'''
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression # Our ML model
from sklearn.preprocessing import LabelEncoder # Preprocess to get float

def read_data_frame(filepath):
    """
    Read CSV
    :type filepath: str
    :rtype: dataframe
    """
    df = pd.read_csv(filepath)
    return df
    
def analysis(df):
    """
    Print data analysis of a given frame
    :type df: data frame
    """
    print("\n--------------------------------------------------------")
    print("What all columns/features we have?")
    print("---------------------------------------------------------")
    print(df.columns)
    print("\n---------------------------------------------------------")
    print("Missing feature/value?")
    print("---------------------------------------------------------")
    print(df.columns[df.isnull().any()])
    print(df.isnull().sum(axis=0))
    print("\n---------------------------------------------------------")
    print("Count per feature/column\n")
    print("---------------------------------------------------------")
    print(df.count())
    print("---------------------------------------------------------")
    print("Describe ...")
    print("---------------------------------------------------------")
    print(df.head())

def preprocess_df(df):
    """
    Preprocess data
    :type df: data frame
    """
    from geopy.distance import vincenty
    # Add new column - distance
    d = lambda row: vincenty(ast.literal_eval(row.POLYLINE)[0], ast.literal_eval(row.POLYLINE)[-1]).miles if not row.MISSING_DATA and bool(ast.literal_eval(row.POLYLINE)) > 0 else 0
    df['distance'] = df.apply(distance = lambda row: vincenty(ast.literal_eval(row.POLYLINE)[0], ast.literal_eval(row.POLYLINE)[-1]))

    # 
        

def main(train_path='../input/train.csv'):
    df = read_data_frame(train_path)
    analysis(df)
    df = preprocess_df(df)
    print("\nAfter preprocessing...")
    print(df.count(),"\n")

main()
