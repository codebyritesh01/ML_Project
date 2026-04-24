import os
import sys 
import dill 
import numpy as np 
import pandas as pd

from src.exception import CustomeException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_models = {}

        for name, model in models.items():
            para = param[name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            report[name] = r2_score(y_test, y_test_pred)
            best_models[name] = best_model   # ✔ MUST be this

        return report, best_models

    except Exception as e:
        raise CustomeException(e, sys)