from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import sys

try:
    filepath = sys.argv[1]
except IndexError:
    filepath = r"C:\Users\Sanket Goel\Downloads\Manish_LR.xlsx"

data = pd.read_excel(filepath)
x = np.array(data['ECL Intensity']).reshape(-1,1)
y = np.array(data['Glucsoe (mM)'])

model = LinearRegression().fit(x,y)
def predict_y(x):
    return f"Glucose: {(model.intercept_ + model.coef_ * x)[0]} (mM)"

while 1:
    try:
        inx = float(input("Intenisty: "))
        if inx >= x[0] and inx <= x[-1]:
            print(predict_y(inx))
        else:
            print("Not in Linear Range")
    except KeyboardInterrupt:
        break