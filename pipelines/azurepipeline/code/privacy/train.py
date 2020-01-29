import mlflow
import mlflow.sklearn
import sklearn
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
import json
import sys
from dp_lin_reg import DPLinearRegression


from sklearn.datasets import load_iris # iris dataset
X, y = load_iris(return_X_y=True) # get X and y

# separate features
sepal_length = X[:, :1]
sepal_width = X[:, 1:2]
petal_length = X[:, 2:3]
petal_width = X[:, 3:4]

if __name__ == "__main__":
    budget = float(sys.argv[3])

    with mlflow.start_run(run_name="diffpriv_covariance_linreg"):
        # Log mlflow attributes for mlflow UI
        mlflow.log_param("dataset_name", "iris")
        mlflow.log_param("budget", budget)
        mlflow.log_param("x_features", "petal length")
        mlflow.log_param("y_targets", "petal width") 

        schema_dict = {"Database": {
            "dbo": {
                "iris": {
                    "rows": 150,
                    "sepal length (cm)": {
                        "type": "float",
                        "min": 4,
                        "max": 8},
                    "sepal width (cm)": {
                        "type": "float",
                        "min": 2,
                        "max": 5},
                    "petal length (cm)": {
                        "type": "float",
                        "min": 1,
                        "max": 7},
                    "petal width (cm)": {
                        "type": "float",
                        "min": 0,
                        "max": 3}
                        }
                    }
                }
            }

        range_dict = schema_dict["Database"]["dbo"]["iris"]
        data_range = pd.DataFrame([[range_dict[col]["min"], range_dict[col]["max"]] for col in ["petal length (cm)", "petal width (cm)"]], index=["petal length", "petal width"], dtype=float).transpose()
        X = pd.DataFrame(petal_length, columns = ["petal length"])
        y = pd.DataFrame(petal_width, columns = ["petal width"])

        for i in range(20):
            try:
                model = DPLinearRegression().fit(X, y, data_range, budget)
                break
            except:
                pass

        # Save model for access through mlflow ui
        mlflow.sklearn.log_model(model, "model")

        results = {
            "run_id": mlflow.active_run().info.run_id,
            "model_name": "diffpriv_linreg"
        }
        with open("result.json", "w") as stream:
            json.dump(results, stream)
        mlflow.log_artifact("result.json")

        