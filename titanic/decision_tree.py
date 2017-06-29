"""
Jun 28th 2017
Dhara
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
    print("What all columns/features we have?")
    print(df[0])
    print("# of people who survived")
    print(df['survived'].mean())
    print("Let's see, how does data look like for each class?")
    print(df.groupby('pclass').mean())


    print("Ok, missing data?")
    print(df.count())

def main(path='data/train.csv'):
    df = read_data_frame(path)
    analysis(df)

main()
