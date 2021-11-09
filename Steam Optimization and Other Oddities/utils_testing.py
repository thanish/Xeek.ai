import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import pytz
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool, cv

UTC = pytz.utc  
timeZ_Kl = pytz.timezone('Asia/Kolkata')


##################
# Create these empty dataFrame first and then comment them to log the RMSE and thier respective indep.

# lgb_best_params = pd.DataFrame(columns=['Date', 'best_RMSE', 'best_param', 'indep'])
# lgb_best_params.to_csv("../Optuna_logging/lgb_optuna_logging.csv", index=False)

# xgb_best_params = pd.DataFrame(columns=['Date', 'best_RMSE', 'best_param', 'indep'])
# xgb_best_params.to_csv("../Optuna_logging/xgb_optuna_logging.csv", index=False)

# cat_best_params = pd.DataFrame(columns=['Date', 'best_RMSE', 'best_param', 'indep'])
# cat_best_params.to_csv("../Optuna_logging/cat_optuna_logging.csv", index=False)
##################


def optuna_logging(study, model, indep):
    print("")
    print(f"Logging the best params for {model} model")
    
    filename = f"../Optuna_logging/{model}_optuna_logging.csv"
    hist_optuna_log = pd.read_csv(filename)
    
    current_best_params = pd.DataFrame({'Date':datetime.now(timeZ_Kl).strftime('%d-%m-%Y %H:%M:%S'),
                                        'best_RMSE':study.best_trial.values, 
                                        'best_param':str(study.best_params),
                                        'indep':str(indep.tolist())})
    
    updated_best_params_optuna_log = hist_optuna_log.append(current_best_params)
    
    updated_best_params_optuna_log.to_csv(filename, index=False)

    
def train_xgb_model(train_df, indep, target, xgb_params):
    """
    This function is to train K-folds(5 in this case) XGBoost model that takes as input the training dataset, independent features,
    target variable, lightGBM hyper-parameters and return the prediction results for each folds.
    
    Parameters
    ----------
    train_df: pandas DataFrame
        This is the training DataFrame which is used to train 5-fold CV models. The folds are labelled under the 
        column named 'fold'.
    indep: list
        This is the list of independent features that will be used to train the model.
    target: str
        This is the target variable 'PCT_DESAT_TO_ORIG' passed as a string.
    lgb_params: dictionary
        This is a dictionary of the best hyper-parameters required to train the XGBoost model. 
        The best hyper-parameter is obtained after tunning them with different combinations.
    
    Returns
    -------
    fold_iterations: list
        This is a list containing the best iteration when the model training was stopped based on early stopping criteria for each fold. 
        It contains 5 values(5-folds) one for each of the trained folds.
    fold_results: list
        This is a list containing the best RMSE value for each fold when it stopped training.
    lgb_models_fold: dictionary
        This is a dictionary containing the model files. It contains 5 values(5-folds), 1 for each fold.    
    lgb_fold_prediction: dictionary
        This is a dictionary containing the out of fold predictions for each fold i.e. When the key is 1, then it is a prediction 
        for the fold 1 from the model trained on data from rest of the fold(0,2,3,4) 
    """
    
    num_rounds = 100000

    fold_iterations = []
    fold_results = []
    xgb_models_fold = {}
    xgb_fold_prediction = {}

    print("")
    for fold_i in range(0, train_df.fold.max()+1):

        train_fold = train_df[train_df.fold!=fold_i].copy()
        valid_fold = train_df[train_df.fold==fold_i].copy()

        dtrain_local = xgb.DMatrix(data= train_fold[indep] , label=train_fold[target])
        dtest_local = xgb.DMatrix(data= valid_fold[indep] , label=valid_fold[target])

        eval_set = [(dtrain_local,'train'), (dtest_local,'test')]

        np.random.seed(100)
        xgb_model_local = xgb.train(xgb_params,
                                    dtrain_local,
                                    evals = eval_set,
                                    num_boost_round = num_rounds,
                                    verbose_eval = False,
                                    early_stopping_rounds = 50)
        xgb_local_prediction = xgb_model_local.predict(dtest_local)

        # Change predictions <0 to 1 and >1 to 1. Since it's not possible to have values <0 or >1
        xgb_local_prediction = np.where(xgb_local_prediction<0, 0, xgb_local_prediction)
        xgb_local_prediction = np.where(xgb_local_prediction>1, 1, xgb_local_prediction)

        fold_rmse = np.sqrt(mean_squared_error(valid_fold[target], xgb_local_prediction))
        fold_iteration = xgb_model_local.best_iteration
        
        fold_iterations.append(fold_iteration)
        fold_results.append(np.round(fold_rmse, 5))
        xgb_models_fold[fold_i] = xgb_model_local
        xgb_fold_prediction[fold_i] = xgb_local_prediction
        
        print(f"Current fold: {fold_i}, iteration {fold_iteration}, RMSE {fold_rmse}")
    
    return fold_iterations, fold_results, xgb_models_fold, xgb_fold_prediction


def train_cat_model(train_df, indep, target, cat_params):
    """
    This function is to train K-folds(5 in this case) CatBoost model that takes as input the training dataset, independent features,
    target variable, lightGBM hyper-parameters and return the prediction results for each folds.
    
    Parameters
    ----------
    train_df: pandas DataFrame
        This is the training DataFrame which is used to train 5-fold CV models. The folds are labelled under the 
        column named 'fold'.
    indep: list
        This is the list of independent features that will be used to train the model.
    target: str
        This is the target variable 'PCT_DESAT_TO_ORIG' passed as a string.
    lgb_params: dictionary
        This is a dictionary of the best hyper-parameters required to train the CatBoost model. 
        The best hyper-parameter is obtained after tunning them with different combinations.
    
    Returns
    -------
    fold_iterations: list
        This is a list containing the best iteration when the model training was stopped based on early stopping criteria for each fold. 
        It contains 5 values(5-folds) one for each of the trained folds.
    fold_results: list
        This is a list containing the best RMSE value for each fold when it stopped training.
    lgb_models_fold: dictionary
        This is a dictionary containing the model files. It contains 5 values(5-folds), 1 for each fold.    
    lgb_fold_prediction: dictionary
        This is a dictionary containing the out of fold predictions for each fold i.e. When the key is 1, then it is a prediction 
        for the fold 1 from the model trained on data from rest of the fold(0,2,3,4) 
    """
    
    num_rounds = 100000
    
    fold_iterations = []
    fold_results = []
    cat_models_fold = {}
    cat_fold_prediction = {}

    print("")
    for fold_i in range(0, train_df.fold.max()+1):

        train_fold = train_df[train_df.fold!=fold_i].copy()
        valid_fold = train_df[train_df.fold==fold_i].copy()
    
        eval_dataset = Pool(valid_fold[indep], valid_fold[target])

        nrounds = 1000000
        np.random.seed(100)
        cat_local_model = CatBoostRegressor(**cat_params
                                            ,iterations=nrounds
                                            ,early_stopping_rounds=50
                                            ,verbose=0
                                            )

        cat_local_model.fit(train_fold[indep],
                            train_fold[target],
                            eval_set=eval_dataset)
        cat_local_prediction = cat_local_model.predict(valid_fold[indep])
        
        # Change predictions <0 to 1 and >1 to 1. Since it's not possible to have values <0 or >1
        cat_local_prediction = np.where(cat_local_prediction<0, 0, cat_local_prediction)
        cat_local_prediction = np.where(cat_local_prediction>1, 1, cat_local_prediction)

        fold_rmse = np.sqrt(mean_squared_error(valid_fold[target], cat_local_prediction))
        fold_iteration = cat_local_model.best_iteration_
        
        fold_iterations.append(fold_iteration)
        fold_results.append(np.round(fold_rmse, 5))
        cat_models_fold[fold_i] = cat_local_model
        cat_fold_prediction[fold_i] = cat_local_prediction
        
        print(f"Current fold: {fold_i}, iteration {fold_iteration}, RMSE {fold_rmse}")
    
    return fold_iterations, fold_results, cat_models_fold, cat_fold_prediction


def train_lgb_model(train_df, indep, target, lgb_params):
    """
    This function is to train K-folds(5 in this case) LightGBM model that takes as input the training dataset, independent features,
    target variable, lightGBM hyper-parameters and return the prediction results for each folds.
    
    Parameters
    ----------
    train_df: pandas DataFrame
        This is the training DataFrame which is used to train 5-fold CV models. The folds are labelled under the 
        column named 'fold'.
    indep: list
        This is the list of independent features that will be used to train the model.
    target: str
        This is the target variable 'PCT_DESAT_TO_ORIG' passed as a string.
    lgb_params: dictionary
        This is a dictionary of the best hyper-parameters required to train the lightGBM model. 
        The best hyper-parameter is obtained after tunning them with different combinations.
    
    Returns
    -------
    fold_iterations: list
        This is a list containing the best iteration when the model training was stopped based on early stopping criteria for each fold. 
        It contains 5 values(5-folds) one for each of the trained folds.
    fold_results: list
        This is a list containing the best RMSE value for each fold when it stopped training.
    lgb_models_fold: dictionary
        This is a dictionary containing the model files. It contains 5 values(5-folds), 1 for each fold.    
    lgb_fold_prediction: dictionary
        This is a dictionary containing the out of fold predictions for each fold i.e. When the key is 1, then it is a prediction 
        for the fold 1 from the model trained on data from rest of the fold(0,2,3,4) 
    """
    
    num_rounds = 100000
    
    fold_iterations = []
    fold_results = []
    lgb_models_fold = {}
    lgb_fold_prediction = {}

    print("")
    # Loop for each of the fold.
    for fold_i in range(0, train_df.fold.max()+1): 

        train_fold = train_df[train_df.fold!=fold_i].copy()
        valid_fold = train_df[train_df.fold==fold_i].copy()
    
        lgb_train_local = lgb.Dataset(train_fold[indep], train_fold[target], free_raw_data=False)
        lgb_test_local = lgb.Dataset(valid_fold[indep], valid_fold[target],
                                     reference=lgb_train_local,  free_raw_data=False)                             

        np.random.seed(100)
        lgb_model_local = lgb.train(lgb_params,
                                    lgb_train_local,
                                    num_boost_round=num_rounds ,
                                    valid_sets=lgb_test_local,
                                    early_stopping_rounds=50,
                                    verbose_eval=False
                                   )
        lgb_local_prediction = lgb_model_local.predict(valid_fold[indep])

        # Change predictions <0 to 1 and >1 to 1. Since it's not possible to have values <0 or >1
        lgb_local_prediction = np.where(lgb_local_prediction<0, 0, lgb_local_prediction)
        lgb_local_prediction = np.where(lgb_local_prediction>1, 1, lgb_local_prediction)

        fold_rmse = np.sqrt(mean_squared_error(valid_fold[target], lgb_local_prediction))
        fold_iteration = lgb_model_local.best_iteration
        
        fold_iterations.append(fold_iteration)
        fold_results.append(np.round(fold_rmse, 5))
        lgb_models_fold[fold_i] = lgb_model_local
        lgb_fold_prediction[fold_i] = lgb_local_prediction
        
        print(f"Current fold: {fold_i}, iteration {fold_iteration}, RMSE {fold_rmse}")
    
    return fold_iterations, fold_results, lgb_models_fold, lgb_fold_prediction