import os
import pickle
from time import time
from numpy.random import seed

import numpy as np                            
import pandas as pd
import tensorflow

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import log_loss, make_scorer
import xgboost as xgb

def train_logistic_regression(directory_inter,lr_filename,lr_train_time_filename,params,X_train,y_train):
    """ Split the data, and either trains a logistic regression model and saves it or loads a trained one
    Args:
        directory_inter: Directory to save or load a logistic regression model. (str) 
        rf_filename: Filename of logistic regression model to be saved om or loaded from directory_inter. (str)
        rf_train_time_filename: Filename of logistic regression training time to be saved on or loaded from directory_inter. (str)
        params: Logistic regression parameters for hyperparameter tuning. (dict)
        X_train: Features to train on. (dataframe)
        y_train: Label for duplicate questions to train on. (dataframe) 
    Returns:
        lr_cl: Trained logistic regression model. (model)
        train_time: Training time of this model. (float)
    """
    if os.path.exists(os.path.join(directory_inter,lr_filename)):
        lr_cl = pd.read_pickle(os.path.join(directory_inter,lr_filename))
        train_time = pd.read_pickle(os.path.join(directory_inter,lr_train_time_filename))
    else:    
        # Initialize classifier
        lr_cl = LogisticRegression()

        # Grid Search for hyperparamter tuning
        grid_search = GridSearchCV(estimator = lr_cl,
                                   param_grid=params,
                                   scoring = 'neg_log_loss',
                                   cv=3,
                                   verbose=2) 

        grid_search.fit(X_train,y_train)
        
        start = time()
        lr_cl = LogisticRegression(**grid_search.best_params_)
        lr_cl.fit(X_train,y_train)
        end = time()        
        train_time = end - start

        # Save the logistic regression model and train_time
        pickle.dump(lr_cl, open(os.path.join(directory_inter,lr_filename), 'wb'))
        pickle.dump(train_time,open(os.path.join(directory_inter,lr_train_time_filename), 'wb'))
    
    return lr_cl , train_time


def train_neural_network(directory_inter,nn_model_filename,nn_train_time_filename,nn_params,X_train,y_train):
    """ Neural network regression model with hyperparameter tuning and saving the model and scores
    Args:
        directory_inter: Directory to save the model. (str)
        nn_model_filename: Name of the filename for neural network classifier model to be saved. (str)
        nn_train_time_filename: Filename of neural network training time to be saved on or loaded from directory_inter. (str)
        nn_params: Neural network classifier hyperparameters. (dict)
        X_train: A dataframe containing features to train on. (dataframe)
        y_test: A dataframe containing the label to train on. (dataframe)
    Returns:
        nn_cl: Neural network classifier model after training to the best hyperparameters.
        train_time: Training time of this model. (float)
    """
    
    def nn_train(hidden_sizes,dropout_rate,batch_size,random_state):
        seed(random_state)
        tensorflow.random.set_seed(random_state)
        nn_cl = Sequential()
        nn_cl.add(Dense(hidden_sizes[0], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
        nn_cl.add(Dropout(dropout_rate))
        nn_cl.add(Dense(hidden_sizes[1], activation='relu'))
        nn_cl.add(Dropout(dropout_rate))
        nn_cl.add(Dense(1, activation='sigmoid'))
        nn_cl.compile(loss='binary_crossentropy', optimizer='adam')

        return nn_cl
    
    # If the model is already trained, then just load it
    if os.path.exists(os.path.join(directory_inter,nn_model_filename)):
        nn_cl = KerasClassifier(build_fn=nn_train)
        nn_cl.model = load_model(os.path.join(directory_inter,nn_model_filename))
        train_time = pd.read_pickle(os.path.join(directory_inter,nn_train_time_filename))
        return nn_cl.model , train_time

    else:
        score2 = make_scorer(log_loss, greater_is_better=False, needs_proba=True, labels=sorted(np.unique(y_train)))
        nn_cl = KerasClassifier(build_fn=nn_train)
        grid_search = GridSearchCV(estimator = nn_cl,
                                   param_grid=nn_params,
                                   scoring = score2,
                                   cv=3,
                                   n_jobs=-1,
                                   pre_dispatch = '2*n_jobs',
                                   return_train_score = True,
                                   verbose=2) 

        grid_search.fit(X_train,y_train)
        start = time()
        nn_cl = KerasClassifier(build_fn= lambda: nn_train(**grid_search.best_params_))
        nn_cl.fit(X_train,y_train)
        end = time()        
        train_time = end - start

        # Save the neural model and train_time
        nn_cl.model.save(os.path.join(directory_inter,nn_model_filename))
        pickle.dump(train_time,open(os.path.join(directory_inter,nn_train_time_filename), 'wb'))
    
        return nn_cl , train_time
    

def train_random_forest(directory_inter,rf_filename,rf_train_time_filename,params,X_train,y_train):
    """ Split the data, and either trains a random forest model and saves it or loads a trained one
    Args:
        directory_inter: Directory to save or load a random forest model. (str) 
        rf_filename: Filename of random forest model to be saved om or loaded from directory_inter. (str)
        rf_train_time_filename: Filename of random forest training time to be saved on or loaded from directory_inter. (str)
        params: Random forest parameters for hyperparameter tuning. (dict)
        X_train: Features to train on. (dataframe)
        y_train: Label for duplicate questions to train on. (dataframe) 
    Returns:
        rf_cl: Trained random forest model. (model)
        train_time: Training time of this model. (float)
    """
    
    if os.path.exists(os.path.join(directory_inter,rf_filename)):
        rf_cl = pd.read_pickle(os.path.join(directory_inter,rf_filename))
        train_time = pd.read_pickle(os.path.join(directory_inter,rf_train_time_filename))
    else:    
        # Initialize classifier
        rf_cl = RandomForestClassifier()

        # Grid Search for hyperparamter tuning
        grid_search = GridSearchCV(estimator = rf_cl,
                                   param_grid=params,
                                   scoring = 'neg_log_loss',
                                   cv=3,
                                   verbose=2) 

        grid_search.fit(X_train,y_train)
        
        start = time()
        rf_cl = RandomForestClassifier(**grid_search.best_params_)
        rf_cl.fit(X_train,y_train)
        end = time()        
        train_time = end - start

        # Save the random forest model and train_time
        pickle.dump(rf_cl, open(os.path.join(directory_inter,rf_filename), 'wb'))
        pickle.dump(train_time,open(os.path.join(directory_inter,rf_train_time_filename), 'wb'))
    
    return rf_cl , train_time
        

def train_xgboost(directory_inter,xgboost_filename,xgboost_train_time_filename,params,X_train,y_train):
    """ Split the data, and either trains an xgboost model and saves it or loads a trained one
    Args:
        directory_inter: Directory to save or load an xgboost model. (str) 
        xgboost_filename: Filename of xgboost model to be saved on or loaded from directory_inter. (str)
        xgboost_train_time_filename: Filename of xgboost training time to be saved on or loaded from directory_inter. (str)
        params: Xgboost parameters for hyperparameter tuning. (dict)
        X_train: Features to train on. (dataframe)
        y_train: Label for duplicate questions to train on. (dataframe) 
    Returns:
        xgb_cl: Trained xgboost model. (model)
        train_time: Training time of this model. (float)
    """
    
    if os.path.exists(os.path.join(directory_inter,xgboost_filename)):
        xgb_cl = pd.read_pickle(os.path.join(directory_inter,xgboost_filename))
        train_time = pd.read_pickle(os.path.join(directory_inter,xgboost_train_time_filename))
    else:    
        # Initialize classifier
        xgb_cl = xgb.XGBClassifier(eval_metric='logloss')

        # Grid Search for hyperparamter tuning
        grid_search = GridSearchCV(estimator = xgb_cl,
                                   param_grid=params,
                                   scoring = 'neg_log_loss',
                                   cv=3,
                                   verbose=2) 

        grid_search.fit(X_train,y_train)
        
        start = time()
        xgb_cl = xgb.XGBClassifier(eval_metric='logloss',**grid_search.best_params_)
        xgb_cl.fit(X_train,y_train)
        end = time()        
        train_time = end - start

        # Save the xgboost model and train_time
        pickle.dump(xgb_cl, open(os.path.join(directory_inter,xgboost_filename), 'wb'))
        pickle.dump(train_time,open(os.path.join(directory_inter,xgboost_train_time_filename), 'wb'))
    
    return xgb_cl, train_time