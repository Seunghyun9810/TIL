### data load
import pandas as pd

df_train = pd.read_csv('C:/Users/USER/Desktop/workspace/TIL/ML/data/titanic_train.csv')
df_test = pd.read_csv('C:/Users/USER/Desktop/workspace/TIL/ML/data/titanic_test.csv')

''' In order to make the preprocessing faster, I'll concat 2 dataframes.'''
df_concat = pd.concat([df_train, df_test], axis=0, ignore_index=True)

### preprocessing
## 1. check missing values
null_count = df_concat.isnull().sum()
print(null_count)               # output - 'Age':263, 'Fare': 1, 'Cabin': 1014, 'Embarked': 2

'''
    For further machine learning process, 'Age', 'Fare', 'Embarked' columns' data must be kept.
    And since there are not much data, those 3 columns' missing values will be filled with specific value differed by each column's charecteristics.
    But 'Cabin' column doesn't have much impact on this process, so this column will be removed.
'''

# 1-1. drop the 'Cabin' column
df_concat = df_concat.drop(columns=['Cabin'])

# 1-2. fill the missing values in 'Age' column
# 1-2-1. check the statistics
print(df_concat['Age'].describe())      # min: 0.17, max: 80, mean: 29.88 ...

# 1-2-2. check the proportions of each age range(0~80's)
import math

age_range = []

for value in df_concat['Age'].values:
    if math.isnan(value) == True:
        continue
    else:
        value = str(value)[0] + '0'
        age_range.append(value)

age_range_series = pd.Series(age_range)
print(age_range_series.value_counts(normalize=True).sort_values(ascending=False))

'''
    * The proportion of each age range(sorted in descending order, rounded to two decimal places ) *
        - 20's : 0.34
        - 30's : 0.23
        - 10's : 0.15
        - 40's : 0.14
        - 50's : 0.07
        - 60's : 0.04
        - 00's : 0.01
        - 70's : 0.01
        - 80's : 0.00
'''