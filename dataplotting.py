import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

health_data = pd.read_csv("CSVs/data2.csv", header=0, sep=",")

x = health_data["Average_Pulse"]
y = health_data["Calorie_Burnage"]
slope_intercept = np.polyfit(x,y,1)
health_data.plot(x ='Average_Pulse', y='Calorie_Burnage', kind='scatter')
plt.show()