import statsmodels.api as sm
import numpy as np
import pandas as pd
import sys

try:
    filepath = sys.argv[1]
except IndexError:
    filepath = r"C:\Users\Sanket Goel\Downloads\Manish_LR.xlsx"

data = pd.read_excel(filepath)
x = sm.add_constant(np.array(data['ECL Intensity']).reshape(-1,1))
y = np.array(data['Glucsoe (mM)'])

model = sm.OLS(y,x).fit()
def predict_y(inx):
    x_new = sm.add_constant(np.array([inx, inx+1]).reshape(-1,1))
    return f"Glucose: {model.predict(x_new)[0]} mM"


while 1:
    try:
        inx = float(input("Intensity: "))
        if inx <= x[0][-1] and inx >= x[-1][-1]:
            print(predict_y(inx))
        else:
            print("Not in Linear Range")
    except KeyboardInterrupt:
        break
