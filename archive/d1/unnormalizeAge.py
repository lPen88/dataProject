import numpy as np
import pandas as pd

df= pd.read_csv('dataset/heart-attack-risk-prediction-dataset.csv')
df=df.dropna()
df= df.drop(columns=['Diet', 'BMI', 'Heart Attack Risk (Text)', 'Sedentary Hours Per Day', 'Exercise Hours Per Week'])
df = df.rename(columns={'Heart Attack Risk (Binary)': 'Heart Attack Risk'})

min_age = 19
max_age = 82

df['Age'] = df['Age'] * (max_age - min_age) + min_age
#convert result to integer
df['Age'] = df['Age'].astype(int)

df.to_csv('dataset/mainDatasetAgeFix.csv', index=False)