import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('csv/cab_booking.csv')

# Make a coloumn that has all the values set to 61
# df['Max Fitness'] = 61
# Display a 3d Surface plot of the data considering the first column as the x-axis, the second column as the y-axis and the third column as the z-axis using mayplotlib and mark the points with a red dot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]], linewidth=0.2, alpha=0.5)
ax.plot_trisurf(df[df.columns[0]], df[df.columns[1]], df[df.columns[3]], linewidth=0.2, color='red', alpha=0.4)
ax.plot_trisurf(df[df.columns[0]], df[df.columns[1]], df[df.columns[4]], linewidth=0.2, color='green', alpha=0.3)
ax.scatter(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]], c='pink', marker='o')
ax.scatter(df[df.columns[0]], df[df.columns[1]], df[df.columns[3]], c='magenta', marker='o')
ax.scatter(df[df.columns[0]], df[df.columns[1]], df[df.columns[4]], c='yellow', marker='o')
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
ax.set_zlabel("Average Fitness")

#Plot z=61 plane with colour green
# ax.plot_trisurf(df[df.columns[0]], df[df.columns[1]], df[df.columns[4]], linewidth=0.2, color='green', alpha=0.3)

plt.show()


