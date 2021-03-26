import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost 

from sklearn.pipeline import Pipeline
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from functools import partial

from vf_portalytics.tool import squared_error_objective_with_weighting
from vf_portalytics.tool import get_categorical_features
from vf_portalytics.ml_helpers import get_transformer

from splitting import Splitting
from transforms import ClusterTransform


class Hyperparameters(object):
    
    def __init__(self, model):
        """
        
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.model = model
        self.space = {
            'n_estimators' : hp.choice('n_estimators', np.arange(100, 500, 100, dtype=int)),
            'max_depth': hp.choice('max_depth', np.arange(2, 6, dtype = int)),
            'subsample': hp.quniform('subsample', 0.4, 0.8, 0.05), 
            'min_child_weight': hp.quniform ('min_child_weight', 1, 20, 1),
            'gamma' : hp.quniform('gamma', 0.7, 1.0, 0.05), #0.7-1.0
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.4, 0.8, 0.05), 
            'learning_rate' : hp.quniform('learning_rate', 0.001, 0.1, 0.01),
            'transformer_nominal': hp.choice('transformer_nominal', ['TargetEncoder', 'JamesSteinEncoder']),
            'transformer_ordinal': hp.choice('transformer_ordinal', ['OrdinalEncoder']),
            'under_predict_weight': hp.choice('under_predict_weight', [2.0, 2.5, 3.0]),
            'reg_alpha' : hp.quniform('reg_alpha', 0.5, 1.0, 0.05), 
            'reg_lambda' : hp.quniform('reg_lambda', 1.0, 1.5, 0.05)
        }
        self.seed = 0
        self.max_evals = self.model.max_evals
        self.target_ratio_val = self.model.target_ratio_val or 0.30
        
    def score(self, params, train_x_group, train_y_group, val_x_group, val_y_group):

        categorical_features = get_categorical_features(data=train_x_group)

        # preprocess ordinals
        transformer_ordinal = get_transformer('OrdinalEncoder')
        gp_ordinal = [feature for feature in categorical_features if feature in self.model.ordinal_features]
        transformer_ordinal.cols = gp_ordinal

        # preprocess nominals
        transformer_nominal = get_transformer('TargetEncoder')
        gp_nominals = [feature for feature in categorical_features \
                       if feature in self.model.nominal_features or feature not in gp_ordinal]
        transformer_nominal.cols = gp_nominals
        assert set(gp_nominals + gp_ordinal) == set(categorical_features)

        gbm_model = xgboost.XGBRegressor(n_estimators = params['n_estimators'], 
                                         objective = partial(squared_error_objective_with_weighting, 
                                                             under_predict_weight=params['under_predict_weight']), 
                                         max_depth = params['max_depth'],
                                         subsample = params['subsample'],
                                         min_child_weight = params['min_child_weight'],
                                         gamma = params['gamma'],
                                         colsample_bytree = params['colsample_bytree'],
                                         learning_rate = params['learning_rate'],
                                         reg_alpha = params['reg_alpha'],
                                         reg_lambda = params['reg_lambda'],  
                                         n_jobs = 8,
                                         seed = 1234,
                                         silent=True)

        pipeline = Pipeline([('transformer_ordinal', transformer_ordinal), 
                         ('transformer_nominal', transformer_nominal), 
                         ('estimator', gbm_model)])

        pipe_trf = Pipeline(pipeline.steps[:-1])
        pipe_trf = pipe_trf.fit(train_x_group, train_y_group)
        eval_set = [(pipe_trf.transform(train_x_group), train_y_group), (pipe_trf.transform(val_x_group), val_y_group)]
        eval_metric = ["mae"]

        #Use CV
    #     score = cross_val_score(pipeline, train_x_group, train_y_group, cv=KFold(n_splits=5, random_state=9876), 
    #                         scoring='neg_mean_squared_error')
    #     loss = np.abs(np.mean(score))
    #     n_estimators = params['n_estimators'] 

        #Use validation performance and early stopping
        pipeline.fit(train_x_group, train_y_group, 
                     estimator__early_stopping_rounds=20, 
                     estimator__eval_set=eval_set, 
                     estimator__eval_metric=eval_metric,
                     estimator__verbose=False)
        n_estimators =  pipeline.named_steps['estimator'].best_ntree_limit
        params['n_estimators'] = n_estimators
        evals_result = pipeline.named_steps['estimator'].evals_result()
        loss = evals_result['validation_1'][eval_metric[0]][n_estimators-1]

        return {'loss' : loss, 'status' : STATUS_OK, 'n_estimators': n_estimators}
    
    def optimize(self, space, train_x_group, train_y_group, val_x_group, val_y_group, gp_key):
        trials = Trials()
        fmin_objective = partial(self.score, train_x_group=train_x_group, train_y_group=train_y_group, 
                                 val_x_group=val_x_group, val_y_group=val_y_group)

        best = fmin(fn=fmin_objective, 
                    space=space, 
                    algo=tpe.suggest, 
                    max_evals=self.max_evals, #What value is optimal?
                    trials=trials
                   )
        return space_eval(space, best), trials
    
    def find_params(self):
        """
        
        """
        # Making one split
        splitting = Splitting(splits=None, data=self.model.data,
                              number_tests=1, target_ratio_test=self.target_ratio_val)
        
        train, val = splitting.get_split(0, self.model.data)
        
        # Exclude products which exists only in test
        # test = test[test['original_pid'].isin(pd.concat([train.original_pid, val.original_pid]))]
        # print("df: {0}, train: {1}, val: {2}, test: {3}".format(df.shape[0], train.shape[0], val.shape[0], test.shape[0]))
        # val = val[val['original_pid'].isin(train.original_pid)]
        print("data: {0}, train: {1}, val: {2}".format(self.model.data.shape[0], train.shape[0], val.shape[0]))

        train_x = train.loc[:, train.columns != self.model.target]
        train_y = train.loc[:, train.columns == self.model.target]

        val_x = val.loc[:, val.columns != self.model.target]
        val_y = val.loc[:, val.columns == self.model.target]

        train_y = train_y.squeeze()
        val_y = val_y.squeeze()
        
        # Create cluster according to cat_feature
        train_x = ClusterTransform(self.model.cat_feature).transform(train_x)

        groups = train_x.groupby('cluster')
        params = {}
        for gp_key, group in groups:
            print('Checking ' + str(gp_key) + ' ...')
            # keep only the most important features
            train_x_group = group[list(self.model.selected_features[gp_key])]
            train_y_group = train_y[train_x_group.index]
            # validation set
            val_x_group = val_x[val_x['cluster']==gp_key]
            # Remove duped rows (possibly duplication has been provided)
            val_x_group = val_x_group.drop('cluster', axis=1).drop_duplicates()
            val_x_group = val_x_group[list(self.model.selected_features[gp_key])]
            val_y_group = val_y[val_x_group.index]
            # find best parameters for each model-group

            best_params, trials = self.optimize(self.space, 
                                           train_x_group, train_y_group, 
                                           val_x_group, val_y_group, 
                                           gp_key)
            params[gp_key] = best_params
            params[gp_key]['n_estimators'] = trials.best_trial['result']['n_estimators']

        self.losses = trials.losses()
        self.params = params
        
    def plot_losses(self):
        """Plot losses with trials object"""
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.rcParams['figure.dpi'] = 100
        plt.plot(self.losses)
        plt.xlabel('Trial')
        plt.ylabel('Loss')
        plt.show()