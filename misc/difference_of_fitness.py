import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('csv/fitness.csv')

# Label 1st column as N, 2nd column as IT and rest all as Trial 1, Trial 2, etc.
df.columns = ['N', 'IT'] + ['Trial ' + str(i) for i in range(1, len(df.columns) - 1)]

df['Max'] = df[[col for col in df.columns if 'Trial' in col]].max(axis=1)
df['Min'] = df[[col for col in df.columns if 'Trial' in col]].min(axis=1)
df['Avg'] = df[[col for col in df.columns if 'Trial' in col]].mean(axis=1)
df['Count'] = df[[col for col in df.columns if 'Trial' in col]].ge(0).sum(axis=1)
df['Median']=df[[col for col in df.columns if 'Trial' in col]].median(axis=1)

# Plot a 3d surface plot between N, IT and Count
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_trisurf(df['N'], df['IT'], df['Median'], cmap=plt.cm.coolwarm, linewidth=0.2)
ax.set_xlabel('N')
ax.set_ylabel('IT')
ax.set_zlabel('Median')
plt.colorbar(surf, shrink=0.5, aspect=5)
ax.scatter(df['N'], df['IT'], df['Median'], c='black', marker='o')



df=df[['N', 'IT', 'Max', 'Min', 'Avg', 'Count', 'Median']]

# Export min, max, avg, count, median, n, it to another excel
df.to_csv('csv/fitness_summary.csv', index=False)