# # import os
# # import sys
# # from dataclasses import dataclass

# # from catboost import CatBoostRegressor
# # from sklearn.ensemble import (
# #     AdaBoostRegressor,
# #     GradientBoostingRegressor,
# #     RandomForestRegressor,
# # )
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import r2_score
# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.tree import DecisionTreeRegressor
# # from xgboost import XGBRegressor

# # from src.exception import CustomException
# # from src.logger import logging
# # from src.utils import save_object, evaluate_models

# # @dataclass
# # class ModelTrainerConfig:
# #     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# # class ModelTrainer:
# #     def __init__(self):
# #         self.model_trainer_config = ModelTrainerConfig()

# #     def initiate_model_trainer(self, train_array, test_array):
# #         try:
# #             logging.info("Split training and test input data")
# #             X_train, y_train, X_test, y_test = (
# #                 train_array[:, :-1],
# #                 train_array[:, -1],
# #                 test_array[:, :-1],
# #                 test_array[:, -1]
# #             )
# #             models = {
# #                 "Random Forest": RandomForestRegressor(),
# #                 "Decision Tree": DecisionTreeRegressor(),
# #                 "Gradient Boosting": GradientBoostingRegressor(),
# #                 "Linear Regression": LinearRegression(),
# #                 "XGBRegressor": XGBRegressor(),
# #                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
# #                 "AdaBoost Regressor": AdaBoostRegressor(),
# #             }
            
# #             model_report = evaluate_models(X_train, y_train, X_test, y_test, models)
            
# #             # Initialize variables to store the best model and score
# #             best_model_name = None
# #             best_model_score = -1
            
# #             # Iterate over the model report and find the best model
# #             for model_name, scores in model_report.items():
# #                 if scores["test_score"] > best_model_score:
# #                     best_model_name = model_name
# #                     best_model_score = scores["test_score"]
# #             print(best_model_score)
            
# #             if best_model_name is None:
# #                 raise CustomException("No best model found")
            
# #             logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

# #             best_model = models[best_model_name]

# #             save_object(
# #                 file_path=self.model_trainer_config.trained_model_file_path,
# #                 obj=best_model
# #             )

# #             predicted = best_model.predict(X_test)

# #             r2_square = r2_score(y_test, predicted)
# #             return r2_square
            
# #         except Exception as e:
# #             raise CustomException(e, sys)
# import os
# import sys
# from dataclasses import dataclass
# from sklearn.model_selection import GridSearchCV

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object, evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1]
#             )
            
#             # Define a dictionary of hyperparameters to search over
#             param_grid = {
#                 'n_estimators': [50, 100, 200],
#                 'max_depth': [None, 10, 20, 30],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'max_features': ['auto', 'sqrt', 'log2']
#             }
            
#             # Initialize the RandomForestRegressor
#             model = RandomForestRegressor()
            
#             # Create the GridSearchCV object
#             grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            
#             # Fit the model to the data, this will perform hyperparameter tuning
#             grid_search.fit(X_train, y_train)
            
#             # Get the best model with tuned hyperparameters
#             best_model = grid_search.best_estimator_
            
#             logging.info(f"Best model with hyperparameter tuning: {best_model}")
            
#             # Save the best model
#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted = best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            
#         except Exception as e:
#             raise CustomException(e, sys)

import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define the hyperparameter grid with valid values for max_features
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]  # Valid values for max_features
            }

            # Create a RandomForestRegressor
            rf_regressor = RandomForestRegressor()

            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, 
                                       scoring='r2', cv=5, n_jobs=-1)

            # Fit the model with hyperparameter tuning
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            logging.info(f"Best model found with hyperparameter tuning: {best_model}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model
            predicted = best_model.predict(X_test)

            # Calculate R-squared score
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
