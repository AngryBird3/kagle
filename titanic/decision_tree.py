"""
Jun 28th 2017
Dhara

Resource: https://blog.socialcops.com/engineering/machine-learning-python/
"""

import pandas as pd

'''
@param filepath
@return dataframe
'''
def read_data_frame(filepath): 
    df = pd.read_csv(filepath)
    return df

'''
@param dataframe
'''
def analysis(df):
    print("\n--------------------------------------------------------")
    print("What all columns/features we have?")
    print("---------------------------------------------------------")
    print(df.columns)
    print("\n---------------------------------------------------------")
    print("# of people who survived")
    print("---------------------------------------------------------")
    print(df['Survived'].mean())
    print("\n---------------------------------------------------------")
    print("Let's see, how does data look like for each class?")
    print("---------------------------------------------------------")
    print(df.groupby('Pclass').mean())


    print("Ok, missing data?")
    print(df.count())

def preprocess_df(df):
    '''
    Data looks likes this for each column
    PassengerId    891 - ID
    Survived       891 - 0 = No, 1 = Yes
    Pclass         891 - Passenger class(1 = first, 2 = second, 3 = third)
    Name           891 - Name
    Sex            891 - Sex
    Age            714 - Age
    SibSp          891 - # of siblings/spouces aboard
    Parch          891 - # of parents/children aboard
    Ticket         891 - Ticket #
    Fare           891 - Passenger fare
    Cabin          204 - Cabin
    Embarked       889 = Port of embarkation(C = Cherbourg, Q = Queentown, S = Southampton)
    dtype: int64
    '''
    processed_df = df.copy()
    processed_df.drop(['Name', 'Ticket', 'cabin'])

def main(path='data/train.csv'):
    df = read_data_frame(path)
    analysis(df)

main()
