import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../csv/online_genetic_1000.csv')

# drop last column
df = df.drop(df.columns[-1], axis=1)

# find max along each row from 1st column to last column and set it as the last column
df['max'] = df.iloc[:, 1:].max(axis=1)

# find variance along each row from 1st column to last column and set it as the last column
df['var'] = df.iloc[:, 1:].var(axis=1)

print(df)
