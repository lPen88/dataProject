import numpy as np
import pandas as pd

df= pd.read_csv('dataset/heart-attack-risk-prediction-dataset.csv')
df=df.dropna()
df= df.drop(columns=['Diet', 'BMI', 'Heart Attack Risk (Text)', 'Sedentary Hours Per Day', 'Exercise Hours Per Week'])
df = df.rename(columns={'Heart Attack Risk (Binary)': 'Heart Attack Risk'})
df.to_csv('dataset/mainDataset2.csv', index=False)