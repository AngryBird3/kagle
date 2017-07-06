"""
Jun 28th 2017
Dhara

Resource: https://blog.socialcops.com/engineering/machine-learning-python/
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

'''
@param filepath
@return dataframe
'''
def read_data_frame(filepath): 
    df = pd.read_csv(filepath)
    return df

'''
@param dataframe
Does bunch of analysis
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


    print("Ok, missing data?\n")
    print(df.count())

'''
Preprocess data
@param df dataframe
@returns dataframe
'''
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
    #Dropping unwanted/not-needed columns
    processed_df = df.copy()
    processed_df = processed_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Fare'], axis=1)
    #print("After dropping name, ticket, cabin, passengerId...")
    #print(processed_df.count())

    #Sex has male/female lable, I want to make it 1=male and 0=female
    le = LabelEncoder()
    processed_df['Sex'] = le.fit_transform(processed_df['Sex'])
    #print("Sex classes: ", le.classes_)
    #Samething for embarked, but before that fill "NA" for missing values
    processed_df['Embarked'] = processed_df['Embarked'].fillna("NA")
    processed_df['Embarked'] = le.fit_transform(processed_df['Embarked'])

    processed_df['Age'] = processed_df['Age'].fillna(0)
    processed_df['Age'].fillna(processed_df['Age'].mean(), inplace=True)
    print("Embarked classes: ", le.classes_)
    return processed_df

'''
Trains Decision Tree classfier using given dataframe
@param df dataframe
@returns trained classifier
'''
def train_decision_tree(df):
    classifier = DecisionTreeClassifier()
    #FWIW axis=1 means column
    X = df.drop(['Survived'], axis=1).values
    Y = df['Survived'].values
    classifier = classifier.fit(X, Y)
    return classifier

def main(train_path='data/train.csv', test_path='data/test.csv'):
    df = read_data_frame(train_path)
    analysis(df)
    df = preprocess_df(df)
    print("\nAfter preprocessing...")
    print(df.count(),"\n")
    classifier = train_decision_tree(df)

    test_df = read_data_frame(test_path)
    test_df = preprocess_df(test_df)
    print("\nAfter preprocessing...")
    print(test_df.count(),"\n")

    predictions = classifier.predict(test_df)
    test_df = read_data_frame(test_path)
    result_df = test_df[['PassengerId']].copy()
    result_df['Survived'] = pd.Series(predictions, index=result_df.index)
    print(result_df.columns)
    result_df.to_csv('prediction1.csv', index=False)
main()
