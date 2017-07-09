import numpy as np # linear algebra
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
    #print("\n--------------------------------------------------------")
    #print("What all columns/features we have?")
    #print("---------------------------------------------------------")
    #print(df.columns)
    print("\n---------------------------------------------------------")
    print("Missing feature/value?")
    print("---------------------------------------------------------")
    #print(df.columns[df.isnull().any()])
    print(df.isnull().sum(axis=0))
    #print("---------------------------------------------------------")
    #print("Describe ...")
    #print("---------------------------------------------------------")
    #print(df.head())

def preprocess_df(df):
    """
    Preprocess data
    :type df: data frame
    """
    # Looks like columns (PoolQC, Fence, MiscFeature, Alley) - has too many null values
    # I'll just drop them
    df = df.drop(['Id', 'PoolQC', 'Fence', 'MiscFeature', 'Alley'], axis=1)
    
    #Alright let's fill remaining value with mean
    df = df.fillna(df.mean())
    print(df.columns[df.isnull().any()])
    #df = pd.get_dummies(df) don't use this, it creates new column based on ALL classification in data and value with be a number, well then for test data we might end up having different # of columns
    
    
    le = LabelEncoder()
    df['MSZoning'] = df['MSZoning'].fillna('NA')
    df['MSZoning'] = le.fit_transform(df['MSZoning'])
    
    df['Street'] = le.fit_transform(df['Street'])
    df['LotShape'] = le.fit_transform(df['LotShape'])
    df['LandContour'] = le.fit_transform(df['LandContour'])
    
    df['Utilities'] = df['Utilities'].fillna('NA')
    df['Utilities'] = le.fit_transform(df['Utilities'])
    
    df['LotConfig'] = le.fit_transform(df['LotConfig'])
    df['LandSlope'] = le.fit_transform(df['LandSlope'])
    df['Neighborhood'] = le.fit_transform(df['Neighborhood'])
    df['Condition1'] = le.fit_transform(df['Condition1'])
    df['Condition2'] = le.fit_transform(df['Condition2'])
    df['BldgType'] = le.fit_transform(df['BldgType'])
    df['HouseStyle'] = le.fit_transform(df['HouseStyle'])
    df['RoofStyle'] = le.fit_transform(df['RoofStyle'])
    df['RoofMatl'] = le.fit_transform(df['RoofMatl'])
    
    df['Exterior1st'] = df['Exterior1st'].fillna('NA')
    df['Exterior1st'] = le.fit_transform(df['Exterior1st'])
    
    df['Exterior2nd'] = df['Exterior2nd'].fillna('NA')
    df['Exterior2nd'] = le.fit_transform(df['Exterior2nd'])

    df['MasVnrType'] = df['MasVnrType'].fillna('NA')
    df['MasVnrType'] = le.fit_transform(df['MasVnrType'])
    
    df['ExterQual'] = le.fit_transform(df['ExterQual'])
    df['ExterCond'] = le.fit_transform(df['ExterCond'])
    df['Foundation'] = le.fit_transform(df['Foundation'])
    
    df['BsmtQual'] = df['BsmtQual'].fillna('NA')
    df['BsmtQual'] = le.fit_transform(df['BsmtQual'])
    
    df['BsmtCond'] = df['BsmtCond'].fillna('NA')
    df['BsmtCond'] = le.fit_transform(df['BsmtCond'])
    
    df['BsmtExposure'] = df['BsmtExposure'].fillna('NA')
    df['BsmtExposure'] = le.fit_transform(df['BsmtExposure'])
    
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NA')
    df['BsmtFinType1'] = le.fit_transform(df['BsmtFinType1'])
    
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NA')
    df['BsmtFinType2'] = le.fit_transform(df['BsmtFinType2'])
    
    df['Heating'] = le.fit_transform(df['Heating'])
    df['HeatingQC'] = le.fit_transform(df['HeatingQC'])
    df['CentralAir'] = le.fit_transform(df['CentralAir'])
    
    df['Electrical'] = df['Electrical'].fillna('NA')
    df['Electrical'] = le.fit_transform(df['Electrical'])
    
    df['KitchenQual'] = df['KitchenQual'].fillna('NA')
    df['KitchenQual'] = le.fit_transform(df['KitchenQual'])
    
    df['Functional'] = df['Functional'].fillna('NA')
    df['Functional'] = le.fit_transform(df['Functional'])
    
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
    df['FireplaceQu'] = le.fit_transform(df['FireplaceQu'])
    
    df['GarageType'] = df['GarageType'].fillna('NA')
    df['GarageType'] = le.fit_transform(df['GarageType'])
    
    df['GarageFinish'] = df['GarageFinish'].fillna('NA')
    df['GarageFinish'] = le.fit_transform(df['GarageFinish'])
    
    df['GarageQual'] = df['GarageQual'].fillna('NA')
    df['GarageQual'] = le.fit_transform(df['GarageQual'])
    
    df['GarageCond'] = df['GarageCond'].fillna('NA')
    df['GarageCond'] = le.fit_transform(df['GarageCond'])
    
    df['PavedDrive'] = le.fit_transform(df['PavedDrive'])
    
    df['SaleType'] = df['SaleType'].fillna('NA')
    df['SaleType'] = le.fit_transform(df['SaleType'])
    
    df['SaleCondition'] = le.fit_transform(df['SaleCondition'])

    return df
    
import numpy as np
def train(df):
    lr = LinearRegression()
    X = df.drop(['SalePrice'], axis=1).values
    Y = np.log(df['SalePrice'].values)
    lr = lr.fit(X, Y)
    return lr
    
def main(train_path='../input/train.csv', test_path='../input/test.csv'):
    train_df = read_data_frame(train_path)
    test_df = read_data_frame(test_path)
    #analysis(train_df)
    #analysis(test_df)


    train_df = preprocess_df(train_df)
    lr = train(train_df)
    
    test_df = preprocess_df(test_df)
    #print(train_df.shape)
    #print(test_df.shape)
    
    predictions = lr.predict(test_df)
    test_df = read_data_frame(test_path)
    result_df = test_df[['Id']].copy()
    result_df['SalePrice'] = pd.Series(np.exp(predictions), index=result_df.index)
    print(result_df.columns)
    result_df.to_csv('prediction2.csv', index=False)	    
    
main()

