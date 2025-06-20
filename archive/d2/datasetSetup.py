import pandas as pd

df= pd.read_csv('dataset/dataset_l.csv')
df=df.dropna()
df= df.drop(columns=['Patient ID', 'Physical Activity Days Per Week'])

df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')

df=df.drop(columns=['Blood Pressure'])

df.to_csv('dataset/dataset_l_fix.csv', index=False)