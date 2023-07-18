import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

from sklearn.linear_model import LinearRegression,RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = { 
    #Add required models here
    "Linear Regression" : LinearRegression(),
    "Huber Regression": HuberRegressor(),
    "KNeighbor": KNeighborsRegressor(),
    "Support Vector Machine Regression" : SVR(),
    "Decision Tree":DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "AdaBoost" : AdaBoostRegressor(),
    "Gradient Boost": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "RANSAC Regression": RANSACRegressor(),
    "Theil-Sen Regression": TheilSenRegressor()
}

model_results = {}
   
def errorMetricsCSV(path):
    path = os.path.join(parentPath, 'error-metrics.csv')
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'R2 Score', 'MAE', 'MSE', 'RMSE'])
        for model_name, model_result in model_results.items():
            r2score = model_result["R2 Score"]
            mae = model_result["MAE"]
            mse = model_result["MSE"]
            rmse = model_result["RMSE"]
            writer.writerow([model_name, r2score, mae, mse, rmse])


def putDataInCSV(path):
    path = os.path.join(parentPath, 'xy-data.csv')
    y_preds = {}
    y_preds[f'{x_label}'] = [float(str(x).replace('[', '').replace(']', '')) for x in x_test.to_numpy().tolist()]
    y_preds[f'Original {y_label}'] = y_test.tolist()
    for model_name, model_result in model_results.items():
        y_preds[model_name] = model_result["Predicted Y"].tolist()

    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(y_preds.keys())
        for i in range(len(y_preds[list(y_preds.keys())[0]])):
            row = []
            for name, val in y_preds.items():
                n_val = val[i]
                row.append(n_val)
            writer.writerow(row)
    
def one_has_all():
    plt.figure(figsize=((len(models)*1.5)//1, (len(models)*1.75)//1))
    for i, (model_name, model_result) in enumerate(model_results.items()):
        if len(models) % 3 ==0:
            row_div, cols, add = 3, 3, 0
        elif len(models)%2 == 0:
            row_div, cols,add = 2,2, 0
        elif len(models)%5 == 0:
            row_div, cols,add = 5,5, 0 
        else:
            row_div,cols,add = 3, 3, 1
        plt.subplot((len(models)//row_div)+add, cols, i+1)
        plotSingleModel(model_name)
    plt.savefig(os.path.join(parentPath, f'models.jpg'))

def singlePlots():
    for model_name, model_result in model_results.items():
        plotSingleModel(model_name, save=True)

def plotSingleModel(model_name,save=False):
    model_result = model_results[model_name]
    y_pred_model = model_result["Predicted Y"]
    if save:
        plt.figure(figsize=(10,10))
    plt.scatter(x,y, color="blue")
    plt.scatter(x_test, y_pred_model, color="red")
    plt.legend(["Orginial", model_name])
    plt.title(f"{model_name} Technique")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        save_name = model_name.strip().replace(" ", "_").strip().lower()
        plt.savefig(os.path.join(parentPath, f"{save_name}.jpg"))
        plt.close()

def all_in_one():
    plt.figure(figsize=(10,10))
    plt.scatter(x,y, color="blue", label="Original")
    for color, (model_name, model_result) in zip(('red', 'green', 'black', 'magenta', 'orange', 'violet', 'brown', 'cyan', 'gray', 'khaki'),model_results.items()):
        plt.scatter(x_test, model_result['Predicted Y'], color=color, label=model_name)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label}")
    plt.savefig(os.path.join(parentPath, f"total.jpg"))

if __name__ == '__main__':
    filepath = r"C:\Users\shashg\Downloads\Abhishek Glucose ML\Glucose_Abhishek_ML - Copy.xlsx"
    parentPath = os.path.split(filepath)[0]
    x_label = "Intensiyt_RLU"
    y_label = "Glucose_mM"
    df = pd.read_excel(filepath)
    x =df.drop([y_label], axis=1)
    y = df[y_label]

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)

    for i, model_name in enumerate(models):
        model = models[model_name]
        model.fit(x_train, y_train)
        y_pred_model = model.predict(x_test)
        model_r2_score = r2_score(y_test, y_pred_model)
        model_mse = mean_squared_error(y_test, y_pred_model)
        model_results[model_name] = {
            "MAE": mean_absolute_error(y_test, y_pred_model),
            "MSE": model_mse,
            "RMSE": np.sqrt(model_mse),
            "R2 Score": model_r2_score,
            "Predicted Y": y_pred_model
            }
    errorMetricsCSV(filepath)
    putDataInCSV(filepath)
    singlePlots()
    one_has_all()
    all_in_one()