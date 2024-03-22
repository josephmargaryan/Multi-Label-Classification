import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
import optuna
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class ModelingBinary:
    """
    Class for fitting multiple models for Binary classification
    using 5 folds with StratifiedKFold.
    Averaging the accuracy score and comparing them in a plot.
    
    Params: 
        models: a dictionary of different ML models
        X: the specified features
        y: the specified targets
    """
    def __init__(self, models: dict, X, y):
        self.models = models
        self.X = X
        self.y = y
        self.avg_accuracy_scores = None
        self.model_accuracy_scores = None
        self.plot = None
    
    def fit_models(self):
        """Function for fitting and scoring each model over 5 folds"""
        # Initialize StratifiedKFold
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        
        # Dictionaries to store average accuracy scores and model accuracy scores
        avg_accuracy_scores = {}
        model_accuracy_scores = {}
        
        # Iterate over each model
        for model_name, model in self.models.items():
            print("Training", model_name)
            
            # List to store accuracy scores for each fold
            fold_accuracy_scores = []
            
            # Iterate over each fold
            for train_index, test_index in kf.split(self.X, self.y):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
        
                # Fit the model on the training data
                model.fit(X_train, y_train)
        
                # Predict classes
                y_preds = model.predict_proba(X_test)[:, 1]
        
                # Calculate accuracy score for the fold
                accuracy_fold = roc_auc_score(y_test, y_preds)
        
                # Append accuracy score for the fold to the list
                fold_accuracy_scores.append(accuracy_fold)
                
            # Calculate average accuracy score across folds for the model
            avg_accuracy_score = np.mean(fold_accuracy_scores)
            
            # Store average accuracy score for the model
            avg_accuracy_scores[model_name] = avg_accuracy_score
            model_accuracy_scores[model_name] = fold_accuracy_scores
        
        self.avg_accuracy_scores = avg_accuracy_scores
        self.model_accuracy_scores = model_accuracy_scores
        self._plot_scores()
    
    def _plot_scores(self):
        """Function responsible for plotting"""
        if self.avg_accuracy_scores is None:
            print("Please fit models first using fit_models method.")
            return
        
        # Plot average accuracy scores for each model
        plt.figure(figsize=(10, 6))
        plt.barh(list(self.avg_accuracy_scores.keys()), list(self.avg_accuracy_scores.values()))
        plt.xlabel('Average Accuracy Score')
        plt.ylabel('Model')
        plt.title('Average Accuracy Score for Different Models')
        
        # Store the plot as an attribute
        self.plot = plt
    
    def show_plot(self):
        """Show the stored plot"""
        if self.plot is None:
            print("No plot available. Please run fit_models method first.")
            return
        self.plot.show()




gb = GradientBoostingClassifier()
mskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Average ROC AUC score: ", np.mean(cross_val_score(gb, X_train, y_train, cv=mskf, scoring = "roc_auc")))



# MultiLabel
xgb = MultiOutputClassifier(XGBClassifier())

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Average ROC AUC score: ", np.mean(cross_val_score(xgb, X_train_scaled, y_train, cv=mskf, scoring = "roc_auc")))




class ModelMulticlass:
    """
    Class to fit multiple XGB models using StratifiedKFold for validation
    Params:
        train: train df
        test: test df
    """
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.model_dict = dict()
        self.test_predict_list = list()
        
    def fit(self, params):
        """
        Function to fit the models
        Params: 
            params: A dictionary of the parameters for the models
        """
        label_column = 'target'  # Assuming the target column is named 'target'
        train_cols = [col for col in self.train.columns.to_list() if col != label_column]
        scores = list()
        
        for i in range(5):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_valid_preds = np.zeros(self.train.shape[0])
                
            for fold, (train_idx, valid_idx) in enumerate(skf.split(self.train[train_cols], self.train[label_column])):
                X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[label_column].iloc[train_idx]
                X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[label_column].iloc[valid_idx]
            
                model = XGBClassifier(random_state=5*i+13*fold, **params) 
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=250, verbose=False)               
                valid_preds = model.predict(X_valid)
                oof_valid_preds[valid_idx] = valid_preds
                test_predict = model.predict(self.test[train_cols])
                self.test_predict_list.append(test_predict)
                score = accuracy_score(y_valid, valid_preds)
                self.model_dict[f'fold_{fold}'] = model                    
            oof_score = accuracy_score(self.train[label_column], oof_valid_preds)
            print(f"The OOF accuracy score for iteration {i+1} is {oof_score}")
            scores.append(oof_score)
        return scores, self.test_predict_list

