import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from vf_portalytics.multi_model import MultiModel
import matplotlib.pyplot as plt

from splitting import Splitting
from utils import suppress_stdout, autolabel


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


class AnalysisMetrics(object):
    
    """
    Use AnalysisMetrics to analyze the model.
    
    """
    
    def __init__(self, model, number_tests=None, confidence_threshold=None,
                 account_test_filter=None, target_ratio_test=0.20):
        """
        - If the model has not been yet analysed (model.evaluation_metrics is None), evaluation of MAPE metric starts
        with all data (historical and future). Parameter number_tests defines number of tests. For each
        test, data randomly splits. Evaluation is needed to define filter that is used to estimate
        credence criteria. Then the calculation of all metrics starts (accounting parameter confidence_threshold).
        - To not apply product filter set confidence_threshold to 0.
        - If number_tests is None, it will be get from the model or take 10.
        - If confidence_threshold is None, it will be get from the model or take 300.
        - If account_test_filter is None, it will be get from the model or take all accounts from the model.data.
        
        """
        if not model.model_trained:
            raise Exception("The model is not trained.")
        self.model = model
        self.full_model = model
        
        self.show_tips = self.model.show_tips
        print("Analysing {} model".format(self.model.model_name))
        
        # Check if evaluation is needed
        self.reevaluate = False
        # Check if recalculation is needed
        self.recalculate = False

        # Get parameters from the model or take the input values
        # number_tests
        if self.model.evaluation_metrics:
            if len(self.model.test_splits) != 0 and isinstance(self.model.test_splits, dict):
                self.number_tests = len(self.model.test_splits)
                print("Model's current number tests: {}".format(self.number_tests))
            if not number_tests is None and number_tests != self.number_tests:
                self.number_tests = number_tests
                print("New number tests: {}".format(self.number_tests))
                self.reevaluate = True
        else:
            self.number_tests = 10 if number_tests is None else number_tests
        # confidence_threshold
        if self.model.evaluation_metrics:
            if not self.model.confidence_threshold is None:
                self.confidence_threshold = self.model.confidence_threshold
                print("Model's current confidence threshold: {}".format(self.confidence_threshold))
            if not confidence_threshold is None and confidence_threshold != self.confidence_threshold:
                self.confidence_threshold = confidence_threshold
                print("New confidence threshold: {}".format(self.confidence_threshold))
                self.recalculate = True
        else:
            self.confidence_threshold = 300 if confidence_threshold is None else confidence_threshold
        # account_test_filter
        if self.model.evaluation_metrics:
            if isinstance(self.model.account_test_filter, list):
                self.account_test_filter = self.model.account_test_filter
                print("Model's current account test filter: {}".format(self.account_test_filter))
            if not account_test_filter is None and account_test_filter != self.account_test_filter:
                self.account_test_filter = account_test_filter
                print("New account test filter: {}".format(self.account_test_filter))
                self.reevaluate = True
        else:
            self.account_test_filter = self.model.data['account_banner'].unique() if account_test_filter is None \
                else account_test_filter
        # target_ratio_test
        if self.model.evaluation_metrics:
            self.target_ratio_test = self.model.target_ratio_test
            print("Model's current target ratio test: {}".format(self.target_ratio_test))
            if not target_ratio_test is None and target_ratio_test != self.target_ratio_test:
                self.target_ratio_test = target_ratio_test
                print("New target ratio test: {}".format(self.target_ratio_test))
                self.reevaluate = True
        else:
            self.target_ratio_test = target_ratio_test
        
        if not self.model.evaluation_metrics:
            print("The model is not evaluated.")
            self.reevaluate = True
        elif self.reevaluate:
            print("The model was evaluated with different parameters.")
            self.model.evaluation_metrics = None
            self.model.confidence_metrics = None
            self.model.test_splits = None
        else:
            print("The model is already evaluated.")
            self.reevaluate = False
            
        if self.reevaluate:
            print("Evaluation is needed, thus, recalculation is needed too")
            self.recalculate = True
        elif self.recalculate:
            print("The model is evaluated, the new threshold is defined")
        else:
            print("Recalculation is not needed")
            self.recalculate = False
                    
        self.test_splitting = Splitting(splits=self.model.test_splits, data=self.model.data,
                                        number_tests=self.number_tests, target_ratio=self.target_ratio_test,
                                        cat_feature = self.model.cat_feature)
        self.__get_unique_index_splits()
        self.__evaluate_model()
        self.__initialize_confidence_filter()
#         self.__initialize_confidence_future_filter()
        self.__calculate_model()
        
        self.evaluation_overalls = self.__overall_metrics('evaluation')
        self.evaluation_accounts = self.__account_metrics('evaluation')
        self.confidence_overalls = self.__overall_metrics('confidence')
        self.confidence_accounts = self.__account_metrics('confidence')
        
        self.__update_model()
        self.__tips_useful_methods()
        
    def __update_model(self):
        self.model.test_splits = self.test_splitting.splits
        self.model.account_test_filter = self.account_test_filter
        self.model.confidence_threshold = self.confidence_threshold
        self.model.evaluation_metrics = self.evaluation_metrics
        self.model.evaluation_deferred_metrics = self.evaluation_deferred_metrics
        self.model.evaluation_future_metrics = self.evaluation_future_metrics
        self.model.confidence_metrics = self.confidence_metrics
        self.model.confidence_deferred_metrics = self.confidence_deferred_metrics
        self.model.confidence_future_metrics = self.confidence_future_metrics
        self.model.target_ratio_test = self.target_ratio_test

    def __get_unique_index_splits(self):
        # Get indexes of unique test splits
        lines = list()
        for key in self.test_splitting.splits.keys():
            split = self.test_splitting.splits[key]
            split.sort()
            lines.append(''.join(split))
        df = pd.DataFrame(lines)
        df = df.drop_duplicates()
        self.unique_index_splits = df.index
        print("Total unique index splits: {}".format(len(self.unique_index_splits)))
    
    def __evaluate_model(self):
        if self.reevaluate:
            print("Evaluating the model with historical data (excluding deferred)...")
            self.__evaluation_metrics()
            print("Evaluating the model with deferred historical data...")
            self.__evaluation_deferred_metrics()
            print("Evaluating the model with future data...")
            self.__evaluation_future_metrics()
        else:
            self.evaluation_metrics = self.model.evaluation_metrics
            self.evaluation_deferred_metrics = self.model.evaluation_deferred_metrics
            self.evaluation_future_metrics = self.model.evaluation_future_metrics
            
    def __initialize_confidence_filter(self):
        """ Find a filter for data"""
        if self.confidence_threshold != 0:
            df = self.__parse_detailed_metrics(['mape_m'], self.evaluation_metrics)
            df = (df.groupby(['account_banner','original_product_dimension_25','original_product_dimension_26'])
                              .median().reset_index())
            filts = dict()
            for i, row in enumerate(df[df['mape_m'] > self.confidence_threshold].itertuples()):
                filt = dict()
                filt['account_banner'] = row[1]
                filt['original_product_dimension_25'] = row[2]
                filt['original_product_dimension_26'] = row[3]
                filts[i] = filt
            self.confidence_filter = filts
        else:
            self.confidence_filter = None
            
    def __calculate_model(self):
        self.confidence_metrics = self.model.confidence_metrics
        self.confidence_deferred_metrics = self.model.confidence_deferred_metrics
        self.confidence_future_metrics = self.model.confidence_future_metrics
        if self.model.confidence_metrics is None or self.recalculate:
            print("Calulation of metrics for historical data (excluding deferred) with 'MAPE' confidence threshold {}." \
                  .format(self.confidence_threshold))
            self.__confidence_metrics()
        if self.model.confidence_deferred_metrics is None or self.recalculate:
            print("Calulation of metrics for deferred data with 'MAPE' confidence threshold {}." \
                  .format(self.confidence_threshold))
            self.__confidence_deferred_metrics()
        if self.model.confidence_future_metrics is None or self.recalculate:
            print("Calulation of metrics for future data with 'MAPE' confidence threshold {}." \
                  .format(self.confidence_threshold))
            self.__confidence_future_metrics()
            
    def __evaluation_metrics(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.__metrics_split(seed, apply_confidence_filter=False))
        self.evaluation_metrics = metrics

    def __evaluation_deferred_metrics(self):
        metrics = self.__deferred_metrics_split(apply_confidence_filter=False)
        self.evaluation_deferred_metrics = metrics

    def __evaluation_future_metrics(self):
        metrics = self.__future_metrics_split(apply_confidence_filter=False)
        self.evaluation_future_metrics = metrics
    
    def __confidence_metrics(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.__metrics_split(seed, apply_confidence_filter=True))
        self.confidence_metrics = metrics

    def __confidence_deferred_metrics(self):
        metrics = self.__deferred_metrics_split(apply_confidence_filter=True)
        self.confidence_deferred_metrics = metrics

    def __confidence_future_metrics(self):
        metrics = self.__future_metrics_split(apply_confidence_filter=True)
        self.confidence_future_metrics = metrics
        
    def __parse_detailed_metrics(self, metric_filter, metrics):
        
        """
        Parses a dictionary that contains metrics to data frame.
        
        Parameters:
        -----------
        metric_filter: list of metric names (e.g. ['bias_m','mape_m'])
        metrics: dictionary, can be evaluation_metrics only
        
        Output: DataFrame, columns: ['account_banner','original_product_dimension_25','original_product_dimension_26']
                                plus metric_filter
                            
        """
        lines = list()
        for i, item in enumerate(metrics):
            if len(metrics) > 1 and i not in self.unique_index_splits:
                # if metrics are related to historical and split not unique
                continue
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    for key_25 in item[key_acc].keys():
                        if isinstance(item[key_acc][key_25], dict):
                            for key_26 in item[key_acc][key_25].keys():
                                if isinstance(item[key_acc][key_25][key_26], dict):
                                    line = list()
                                    line.append(key_acc)
                                    line.append(key_25)
                                    line.append(key_26)
                                    for metric in metric_filter:
                                        line.append(item[key_acc][key_25][key_26].get(metric, None))
                                    lines.append(line)
        df = pd.DataFrame(lines)
        df.columns = ['account_banner','original_product_dimension_25','original_product_dimension_26'] + metric_filter
        return df

    def __metrics_split(self, seed, apply_confidence_filter):

        if seed % 5 == 0:
            print("Test iteration {} of {}".format(seed + 1, self.number_tests))
        
        self.train, self.test = self.test_splitting.get_split(seed, self.model.data)

        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]

        if apply_confidence_filter:
            # Exclude products with large mape metric from test
            self.test = self.__apply_confidence_filter(self.test, self.confidence_filter)
    
        # train in train_set and check the results in test set
        # split data into X, Y
        self.train_x = self.train.loc[:, self.train.columns != self.model.target]
        self.train_y = self.train.loc[:, self.train.columns == self.model.target]

        self.test_x = self.test.loc[:, self.test.columns != self.model.target]
        self.test_y = self.test.loc[:, self.test.columns == self.model.target]

        # transform to Series
        self.train_y = self.train_y.squeeze()
        self.test_y = self.test_y.squeeze()
        
        # Train model and validate performance
        with suppress_stdout():
            self.model.model.fit(self.train_x, self.train_y)
    
        self.pred_train_y = self.model.model.predict(self.train_x)
        self.pred_test_y = self.model.model.predict(self.test_x)

        m = dict()
        metric_filter = ['r2_m', 'r2_h', 'r2_m_h', 'wape_m', 'wape_h', 'wape_m_h', 'mape_m', 'mape_h', 'mape_m_h', \
                         'bias_m', 'bias_h', 'bias_m_h', 'sfa_m', 'sfa_h', 'sfa_m_h']
        
        # Account metrics
        for account in self.test['account_banner'].unique():
            m_acc = dict()
            
            if not apply_confidence_filter:
                # Performs only for evaluation model.
                # Get detailed metrics for a restricted set of metrics.
                for product_25 in self.test[self.test['account_banner']==account]['original_product_dimension_25'].unique():
                    m_25 = dict()

                    for product_26 in self.test[(self.test['account_banner']==account) & \
                                                (self.test['original_product_dimension_25']==product_25)] \
                                            ['original_product_dimension_26'].unique():
                        m_26 = dict()
                        metric_dict = self.__get_metrics_by_account_product25_product26(['mape_m','bias_m','mape_h','bias_h'],
                                                                                        account, product_25, product_26)
                        m_26['mape_m'] = metric_dict['mape_m']
                        m_26['bias_m'] = metric_dict['bias_m']
                        m_26['mape_h'] = metric_dict['mape_h']
                        m_26['bias_h'] = metric_dict['bias_h']
                        m_25[product_26] = m_26

                    m_acc[product_25] = m_25

            metric_dict = self.__get_metrics_by_account(metric_filter, account)

            m_acc['r2_m'] = metric_dict['r2_m']
            m_acc['r2_h'] = metric_dict['r2_h']
            m_acc['r2__m_h'] = metric_dict['r2_m_h']
            m_acc['wape_m'] = metric_dict['wape_m']
            m_acc['wape_h'] = metric_dict['wape_h']
            m_acc['wape__m_h'] = metric_dict['wape_m_h']
            m_acc['mape_m'] = metric_dict['mape_m']
            m_acc['mape_h'] = metric_dict['mape_h']    
            m_acc['mape__m_h'] = metric_dict['mape_m_h']    
            m_acc['bias_m'] = metric_dict['bias_m']
            m_acc['bias_h'] = metric_dict['bias_h']
            m_acc['bias__m_h'] = metric_dict['bias_m_h']
            m_acc['sfa_m'] = metric_dict['sfa_m']
            m_acc['sfa_h'] = metric_dict['sfa_h']
            m_acc['sfa__m_h'] = metric_dict['sfa_m_h']
            m[account] = m_acc

        # Overall metrics
        m['seed'] = seed
        m['train_r2'] = round(r2_score(self.train_y, self.pred_train_y), 2)
        m['train'] = self.train_x.shape[0]
        m['test'] = self.test_x.shape[0]
            
        metric_dict = self.__get_overall_metrics(metric_filter)

        m['r2_m'] = metric_dict['r2_m']
        m['r2_h'] = metric_dict['r2_h']
        m['r2__m_h'] = metric_dict['r2_m_h']
        m['wape_m'] = metric_dict['wape_m']
        m['wape_h'] = metric_dict['wape_h']
        m['wape__m_h'] = metric_dict['wape_m_h']
        m['mape_m'] = metric_dict['mape_m']
        m['mape_h'] = metric_dict['mape_h']    
        m['mape__m_h'] = metric_dict['mape_m_h']    
        m['bias_m'] = metric_dict['bias_m']
        m['bias_h'] = metric_dict['bias_h']
        m['bias__m_h'] = metric_dict['bias_m_h']
        m['sfa_m'] = metric_dict['sfa_m']
        m['sfa_h'] = metric_dict['sfa_h']
        m['sfa__m_h'] = metric_dict['sfa_m_h']

        return m

    def __deferred_metrics_split(self, apply_confidence_filter):
        """
        # To estimate the deferred dataset, we use full_model that was trained for all data (exlusing the deferred part)
        
        """
        self.test = self.full_model.deferred_data.copy()
        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]
        # Apply confidence filter
        if apply_confidence_filter:
            self.test = self.__apply_confidence_filter(self.test, self.confidence_filter)
        self.test_x = self.test.loc[:, self.test.columns != self.full_model.target]
        self.test_y = self.test.loc[:, self.test.columns == self.full_model.target]
        self.pred_test_y = self.full_model.model.predict(self.test_x)

        m = dict()
        # The same metrics as for historic dataset
        metric_filter = ['r2_m', 'r2_h', 'r2_m_h', 'wape_m', 'wape_h', 'wape_m_h', 'mape_m', 'mape_h', 'mape_m_h', \
                         'bias_m', 'bias_h', 'bias_m_h', 'sfa_m', 'sfa_h', 'sfa_m_h']

        for account in self.test['account_banner'].unique():
            m_acc = dict()

            metric_dict = self.__get_metrics_by_account(metric_filter, account)

            # All metrics are renamed due to conflict naming convention with historic metrics
            m_acc['r2_d_m'] = metric_dict['r2_m']
            m_acc['r2_d_h'] = metric_dict['r2_h']
            m_acc['r2__d_m_h'] = metric_dict['r2_m_h']
            m_acc['wape_d_m'] = metric_dict['wape_m']
            m_acc['wape_d_h'] = metric_dict['wape_h']
            m_acc['wape__d_m_h'] = metric_dict['wape_m_h']
            m_acc['mape_d_m'] = metric_dict['mape_m']
            m_acc['mape_d_h'] = metric_dict['mape_h']    
            m_acc['mape__d_m_h'] = metric_dict['mape_m_h']    
            m_acc['bias_d_m'] = metric_dict['bias_m']
            m_acc['bias_d_h'] = metric_dict['bias_h']
            m_acc['bias__d_m_h'] = metric_dict['bias_m_h']
            m_acc['sfa_d_m'] = metric_dict['sfa_m']
            m_acc['sfa_d_h'] = metric_dict['sfa_h']
            m_acc['sfa__d_m_h'] = metric_dict['sfa_m_h']
            
            m[account] = m_acc

        # Overall metrics
        metric_dict = self.__get_overall_metrics(metric_filter)
        
        # All metrics are renamed due to conflict naming convention with historic metrics
        m['r2_d_m'] = metric_dict['r2_m']
        m['r2_d_h'] = metric_dict['r2_h']
        m['r2__d_m_h'] = metric_dict['r2_m_h']
        m['wape_d_m'] = metric_dict['wape_m']
        m['wape_d_h'] = metric_dict['wape_h']
        m['wape__d_m_h'] = metric_dict['wape_m_h']
        m['mape_d_m'] = metric_dict['mape_m']
        m['mape_d_h'] = metric_dict['mape_h']    
        m['mape__d_m_h'] = metric_dict['mape_m_h']    
        m['bias_d_m'] = metric_dict['bias_m']
        m['bias_d_h'] = metric_dict['bias_h']
        m['bias__d_m_h'] = metric_dict['bias_m_h']
        m['sfa_d_m'] = metric_dict['sfa_m']
        m['sfa_d_h'] = metric_dict['sfa_h']
        m['sfa__d_m_h'] = metric_dict['sfa_m_h']

        metrics = list()
        metrics.append(m)
        return metrics
    
    def __future_metrics_split(self, apply_confidence_filter):
        """
        # To estimate the future dataset, we use full_model that was trained for all data (exlusing the deferred part)
        
        """
        self.test = self.full_model.future_data.copy()
        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]
        # Apply product test filter
        if apply_confidence_filter:
            self.test = self.__apply_confidence_filter(self.test, self.confidence_filter)
        self.test_x = self.test.loc[:, self.test.columns != self.full_model.future_target]
        self.test_y = self.test.loc[:, self.test.columns == self.full_model.future_target]
        self.pred_test_y = self.full_model.model.predict(self.test_x)

        m = dict()
        metric_filter = ['r2_f_m_h', 'wape_f_m_h', 'mape_f_m_h', 'bias_f_m_h', 'sfa_f_m_h']

        for account in self.test['account_banner'].unique():
            m_acc = dict()

            metric_dict = self.__get_metrics_by_account(metric_filter, account)
            
            m_acc['r2_f_m_h'] = metric_dict['r2_f_m_h']
            m_acc['wape_f_m_h'] = metric_dict['wape_f_m_h']
            m_acc['mape_f_m_h'] = metric_dict['mape_f_m_h']
            m_acc['bias_f_m_h'] = metric_dict['bias_f_m_h']
            m_acc['sfa_f_m_h'] = metric_dict['sfa_f_m_h']

#             for product_25 in self.test[self.test['account_banner']==account]['original_product_dimension_25'].unique():
#                 m_25 = dict()
#                 metric_dict = \
#                     self.__get_metrics_by_account_product25(metric_filter, account, product_25)
                
#                 m_25['r2_f_m_h'] = metric_dict['r2_f_m_h']
#                 m_25['wape_f_m_h'] = metric_dict['wape_f_m_h']
#                 m_25['mape_f_m_h'] = metric_dict['mape_f_m_h']
#                 m_25['bias_f_m_h'] = metric_dict['bias_f_m_h']
#                 m_25['sfa_f_m_h'] = metric_dict['sfa_f_m_h']

#                 for product_26 in self.test[(self.test['account_banner']==account) & \
#                                             (self.test['original_product_dimension_25']==product_25)] \
#                                         ['original_product_dimension_26'].unique():
#                     m_26 = dict()
#                     metric_dict = \
#                         self.__get_metrics_by_account_product25_product26(metric_filter, account, product_25, product_26)
                    
#                     m_26['r2_f_m_h'] = metric_dict['r2_f_m_h']
#                     m_26['wape_f_m_h'] = metric_dict['wape_f_m_h']
#                     m_26['mape_f_m_h'] = metric_dict['mape_f_m_h']
#                     m_26['bias_f_m_h'] = metric_dict['bias_f_m_h']
#                     m_26['sfa_f_m_h'] = metric_dict['sfa_f_m_h']
#                     m_25[product_26] = m_26

#                 m_acc[product_25] = m_25

            m[account] = m_acc

        # Overall metrics
        metric_dict = self.__get_overall_metrics(metric_filter)

        m['r2_f_m_h'] = metric_dict['r2_f_m_h']
        m['wape_f_m_h'] = metric_dict['wape_f_m_h']
        m['mape_f_m_h'] = metric_dict['mape_f_m_h']
        m['bias_f_m_h'] = metric_dict['bias_f_m_h']
        m['sfa_f_m_h'] = metric_dict['sfa_f_m_h']

        metrics = list()
        metrics.append(m)
        return metrics
    
    def __apply_confidence_filter(self, data, filters):
        df = data.copy()
        if self.confidence_threshold != 0:
            for key in filters.keys():
                df = df[~((df['account_banner']==filters[key]['account_banner']) & \
                          (df['original_product_dimension_25']==filters[key]['original_product_dimension_25']) & \
                          (df['original_product_dimension_26']==filters[key]['original_product_dimension_26']))]
        return df

    def __historic_overall_metrics(self, metrics):
        # Historic Data
        lines = list()
        for i, item in enumerate(metrics):
            if i not in self.unique_index_splits:
                # if metrics are related to split that is not unique
                continue
            line = dict()
            line['seed'] = item['seed']
            line['train_r2'] = item['train_r2']
            line['train'] = item['train']
            line['test'] = item['test']
            line['Overall / r2_m'] = item['r2_m']
            line['Overall / r2_h'] = item['r2_h']
            line['Overall / r2__m_h'] = item['r2__m_h']
            line['Overall / wape_m'] = item['wape_m']
            line['Overall / wape_h'] = item['wape_h']
            line['Overall / wape__m_h'] = item['wape__m_h']
            line['Overall / mape_m'] = item['mape_m']
            line['Overall / mape_h'] = item['mape_h']
            line['Overall / mape__m_h'] = item['mape__m_h']
            line['Overall / bias_m'] = item['bias_m']
            line['Overall / bias_h'] = item['bias_h']
            line['Overall / bias__m_h'] = item['bias__m_h']
            line['Overall / sfa_m'] = item['sfa_m']
            line['Overall / sfa_h'] = item['sfa_h']
            line['Overall / sfa__m_h'] = item['sfa__m_h']
            lines.append(line)
        data = pd.DataFrame(lines)
        data = data[['seed','train_r2','train','test',
                     'Overall / r2_m','Overall / r2_h','Overall / r2__m_h',
                     'Overall / wape_m','Overall / wape_h','Overall / wape__m_h',
                     'Overall / mape_m','Overall / mape_h','Overall / mape__m_h',
                     'Overall / bias_m','Overall / bias_h','Overall / bias__m_h',
                     'Overall / sfa_m','Overall / sfa_h','Overall / sfa__m_h',]]
        return data    
    
    def __deferred_overall_metrics(self, metrics):
        # Deferred data
        lines = list()
        for item in metrics:
            line = dict()
            line['Overall / r2_d_m'] = item['r2_d_m']
            line['Overall / r2_d_h'] = item['r2_d_h']
            line['Overall / r2__d_m_h'] = item['r2__d_m_h']
            line['Overall / wape_d_m'] = item['wape_d_m']
            line['Overall / wape_d_h'] = item['wape_d_h']
            line['Overall / wape__d_m_h'] = item['wape__d_m_h']
            line['Overall / mape_d_m'] = item['mape_d_m']
            line['Overall / mape_d_h'] = item['mape_d_h']
            line['Overall / mape__d_m_h'] = item['mape__d_m_h']
            line['Overall / bias_d_m'] = item['bias_d_m']
            line['Overall / bias_d_h'] = item['bias_d_h']
            line['Overall / bias__d_m_h'] = item['bias__d_m_h']
            line['Overall / sfa_d_m'] = item['sfa_d_m']
            line['Overall / sfa_d_h'] = item['sfa_d_h']
            line['Overall / sfa__d_m_h'] = item['sfa__d_m_h']
            lines.append(line)
        data = pd.DataFrame(lines)
        data = data[['Overall / r2_d_m','Overall / r2_d_h','Overall / r2__d_m_h',
                     'Overall / wape_d_m','Overall / wape_d_h','Overall / wape__d_m_h',
                     'Overall / mape_d_m','Overall / mape_d_h','Overall / mape__d_m_h',
                     'Overall / bias_d_m','Overall / bias_d_h','Overall / bias__d_m_h',
                     'Overall / sfa_d_m','Overall / sfa_d_h','Overall / sfa__d_m_h']]
        return data
    
    def __future_overall_metrics(self, metrics):
        # Future data
        lines = list()
        for item in metrics:
            line = dict()
            line['Overall / r2_f_m_h'] = item['r2_f_m_h']
            line['Overall / wape_f_m_h'] = item['wape_f_m_h']
            line['Overall / mape_f_m_h'] = item['mape_f_m_h']
            line['Overall / bias_f_m_h'] = item['bias_f_m_h']
            line['Overall / sfa_f_m_h'] = item['sfa_f_m_h']
            lines.append(line)
        data = pd.DataFrame(lines)
        data = data[['Overall / r2_f_m_h','Overall / wape_f_m_h','Overall / mape_f_m_h',
                     'Overall / bias_f_m_h','Overall / sfa_f_m_h']]
        return data
    
    def __overall_metrics(self, metric_type):
        """Get overall metrics"""
        if metric_type not in ['evaluation','confidence']:
            metric_type = 'evaluation'
        # Select metrics
        if metric_type == 'evaluation':
            metrics = self.evaluation_metrics
            deferred_metrics = self.evaluation_deferred_metrics
            future_metrics = self.evaluation_future_metrics
        else:
            metrics = self.confidence_metrics
            deferred_metrics = self.confidence_deferred_metrics
            future_metrics = self.confidence_future_metrics
        # Historic test results            
        data = self.__historic_overall_metrics(metrics)
        data = data.drop('seed', axis=1)
        data = data.quantile([0.5]).reset_index(drop=True)
        # Deferred test results
        deferred_data = self.__deferred_overall_metrics(deferred_metrics)
        # Future test results
        future_data = self.__future_overall_metrics(future_metrics)
        # Merge historic, deferred and future test results
        df = pd.concat([data, deferred_data, future_data], axis=1)
        return df
    
    def __historic_account_metrics(self, metrics):
        # Historic Data
        lines = list()
        for i, item in enumerate(metrics):
            if i not in self.unique_index_splits:
                # if metrics are related to the split that is not unique
                continue
            line = dict()
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    line[key_acc + ' / r2_m'] = item[key_acc]['r2_m']
                    line[key_acc + ' / r2_h'] = item[key_acc]['r2_h']
                    line[key_acc + ' / r2__m_h'] = item[key_acc]['r2__m_h']
                    line[key_acc + ' / wape_m'] = item[key_acc]['wape_m']
                    line[key_acc + ' / wape_h'] = item[key_acc]['wape_h']
                    line[key_acc + ' / wape__m_h'] = item[key_acc]['wape__m_h']
                    line[key_acc + ' / mape_m'] = item[key_acc]['mape_m']
                    line[key_acc + ' / mape_h'] = item[key_acc]['mape_h']
                    line[key_acc + ' / mape__m_h'] = item[key_acc]['mape__m_h']
                    line[key_acc + ' / bias_m'] = item[key_acc]['bias_m']
                    line[key_acc + ' / bias_h'] = item[key_acc]['bias_h']
                    line[key_acc + ' / bias__m_h'] = item[key_acc]['bias__m_h']
                    line[key_acc + ' / sfa_m'] = item[key_acc]['sfa_m']
                    line[key_acc + ' / sfa_h'] = item[key_acc]['sfa_h']
                    line[key_acc + ' / sfa__m_h'] = item[key_acc]['sfa__m_h']
            lines.append(line)
        data = pd.DataFrame(lines)
        return data
    
    def __deferred_account_metrics(self, metrics):
        # Deferred data
        lines = list()
        for item in metrics:
            line = dict()
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    line[key_acc + ' / r2_d_m'] = item[key_acc]['r2_d_m']
                    line[key_acc + ' / r2_d_h'] = item[key_acc]['r2_d_h']
                    line[key_acc + ' / r2__d_m_h'] = item[key_acc]['r2__d_m_h']
                    line[key_acc + ' / wape_d_m'] = item[key_acc]['wape_d_m']
                    line[key_acc + ' / wape_d_h'] = item[key_acc]['wape_d_h']
                    line[key_acc + ' / wape__d_m_h'] = item[key_acc]['wape__d_m_h']
                    line[key_acc + ' / mape_d_m'] = item[key_acc]['mape_d_m']
                    line[key_acc + ' / mape_d_h'] = item[key_acc]['mape_d_h']
                    line[key_acc + ' / mape__d_m_h'] = item[key_acc]['mape__d_m_h']
                    line[key_acc + ' / bias_d_m'] = item[key_acc]['bias_d_m']
                    line[key_acc + ' / bias_d_h'] = item[key_acc]['bias_d_h']
                    line[key_acc + ' / bias__d_m_h'] = item[key_acc]['bias__d_m_h']
                    line[key_acc + ' / sfa_d_m'] = item[key_acc]['sfa_d_m']
                    line[key_acc + ' / sfa_d_h'] = item[key_acc]['sfa_d_h']
                    line[key_acc + ' / sfa__d_m_h'] = item[key_acc]['sfa__d_m_h']
            lines.append(line)
        data = pd.DataFrame(lines)
        return data

    def __future_account_metrics(self, metrics):
        # Future data
        lines = list()
        for item in metrics:
            line = dict()
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    line[key_acc + ' / r2_f_m_h'] = item[key_acc]['r2_f_m_h']
                    line[key_acc + ' / wape_f_m_h'] = item[key_acc]['wape_f_m_h']
                    line[key_acc + ' / mape_f_m_h'] = item[key_acc]['mape_f_m_h']
                    line[key_acc + ' / bias_f_m_h'] = item[key_acc]['bias_f_m_h']
                    line[key_acc + ' / sfa_f_m_h'] = item[key_acc]['sfa_f_m_h']
            lines.append(line)
        data = pd.DataFrame(lines)
        return data

    def __account_metrics(self, metric_type):
        """Get metrics by accounts"""
        if metric_type not in ['evaluation','confidence']:
            metric_type = 'evaluation'
        # Select metrics
        if metric_type == 'evaluation':
            metrics = self.evaluation_metrics
            deferred_metrics = self.evaluation_deferred_metrics
            future_metrics = self.evaluation_future_metrics
        else:
            metrics = self.confidence_metrics
            deferred_metrics = self.confidence_deferred_metrics
            future_metrics = self.confidence_future_metrics
        # Historic test results
        data = self.__historic_account_metrics(metrics)
        data = data.quantile([0.5]).reset_index(drop=True)
        # Deferred test results
        deferred_data = self.__deferred_account_metrics(deferred_metrics)
        # Future test results
        future_data = self.__future_account_metrics(future_metrics)
        # Merge historic, deferred and future data
        df = pd.concat([data, deferred_data, future_data], axis=1)
        return df
    
    def metrics_median(self, level):
        if level == 'overall':
            return self.overalls.median()
        else:
            return self.accounts.median()
    
    def best_iter(self):
        return np.argmax(self.accounts.loc[:,['5Pyaterochka (X5) / r2_m','Lenta / r2_m',
                                               'Dixy / r2_m','Okey Group / r2_m']].sum(axis=1))
    
    def metric_quantiles(self, metric):
        df = pd.concat([self.overalls, self.accounts], axis=1)
        df = df.drop('seed', axis=1)
        columns = list()
        for column in df.columns:
            if metric + '_' in column:
                columns.append(column)
        return df[columns].quantile([0.25,0.5,0.75], axis=0)
    

        
    def wape_score(self, true, pred):
        """ Weighted absolute percent error """
        return round(100.0 * np.sum(np.abs(true.squeeze() - pred.squeeze())) / np.sum(np.abs(true.squeeze())),2)

    def mape_score(self, true, pred):
        return round(100 * np.sum(np.abs((true.squeeze() - pred.squeeze()) / true.squeeze())) / len(true))

    def bias_score(self, true, pred):
        return round(np.sum(pred.squeeze() - true.squeeze()) / np.sum(true.squeeze()) * 100)

    def sfa_score(self, true, pred):
        """ Weighted average accuracy for which errors in plus-overprediction
        are penalized less than errors in minus-underpredictions."""
        return round(100 * (1. - (np.sum(np.abs(pred.squeeze() - true.squeeze())) / np.sum(pred.squeeze()))), 1)

    def __calculate_metrics(self, metric_filter, filter):
        if filter is not None:
            true = self.test_y.loc[filter].copy()
            pred_m = self.pred_test_y.loc[filter].copy()
            pred_h = self.test.loc[filter]['total_units_2'].copy()
        else:
            true = self.test_y.copy()
            pred_m = self.pred_test_y.copy()
            pred_h = self.test['total_units_2'].copy()
        choices = {
            'r2_m': round(r2_score(true, pred_m), 2),
            'r2_h': round(r2_score(true, pred_h), 2),
            'r2_m_h': round(r2_score(pred_h, pred_m), 2),
            'r2_f_m_h': round(r2_score(true, pred_m), 2),
            'wape_m': self.wape_score(true, pred_m),
            'wape_h': self.wape_score(true, pred_h),
            'wape_m_h': self.wape_score(pred_h, pred_m),
            'wape_f_m_h': self.wape_score(true, pred_m),
            'mape_m': self.mape_score(true, pred_m),
            'mape_h': self.mape_score(true, pred_h),
            'mape_m_h': self.mape_score(pred_h, pred_m),
            'mape_f_m_h': self.mape_score(true, pred_m),
            'bias_m': self.bias_score(true, pred_m),
            'bias_h': self.bias_score(true, pred_h),
            'bias_m_h': self.bias_score(pred_h, pred_m),
            'bias_f_m_h': self.bias_score(true, pred_m),
            'sfa_m': self.sfa_score(true, pred_m),
            'sfa_h': self.sfa_score(true, pred_h),
            'sfa_m_h': self.sfa_score(pred_h, pred_m),
            'sfa_f_m_h': self.sfa_score(true, pred_m),
        }
        result = dict()
        for metric in metric_filter:
            result[metric] = choices.get(metric, 'default')
        return result

    def __get_overall_metrics(self, metric_filter):
        return self.__calculate_metrics(metric_filter, filter=None)

    def __get_metrics_by_account(self, metric_filter, account):
        filter = self.test['account_banner']==account
        return self.__calculate_metrics(metric_filter, filter)

    def __get_metrics_by_account_product25(self, metric_filter, account, product_25):
        filter = (self.test['account_banner']==account) & \
            (self.test['original_product_dimension_25']==product_25)
        return self.__calculate_metrics(metric_filter, filter)

    def __get_metrics_by_account_product25_product26(self, metric_filter, account, product_25, product_26):
        filter = (self.test['account_banner']==account) & \
            (self.test['original_product_dimension_25']==product_25) & \
            (self.test['original_product_dimension_26']==product_26)
        return self.__calculate_metrics(metric_filter, filter)

    def confidence_criteria(self):
        """
        Save Model Confidence Criteria report.
        
        """
        if self.confidence_threshold == 0:
            print("Confidence threshold is not defined.")
            return
        df = self.__parse_detailed_metrics(['mape_m','mape_h'], self.evaluation_metrics)
        df = (df.groupby(['account_banner','original_product_dimension_25','original_product_dimension_26'])
                              .median().reset_index())
        
        df = (df.groupby(['account_banner','original_product_dimension_25','original_product_dimension_26'])
                  .median()
                  .reset_index())

        df.columns = ['Account','Category','Brand','MAPE Model','MAPE Human']
        df['MAPE Model'] = df['MAPE Model'].astype(int)
        df['MAPE Human'] = df['MAPE Human'].astype(int)
        
        report_name = os.path.join(self.model.output_dir, self.model.model_name + '_confidence_criteria.xlsx')
        sort_columns = ['MAPE Model']
                                   
        with pd.ExcelWriter('outputs/' + self.model.model_name + '_confidence_criteria.xlsx') as writer:
            for account in df['Account'].unique():
                df_p = df[df['Account']==account].sort_values(sort_columns).reset_index(drop=True)
                df_p.to_excel(writer, sheet_name=account)
                                   
        print("Model Confidence Criteria report is saved to {}".format(report_name))
            
    def __get_supported_metrics(self, metric_filter):
        supported_metrics = list(['r2','wape','mape','bias','sfa'])
        if metric_filter is None or not isinstance(metric_filter, list):
            cleared_metic_filter = supported_metrics
        else:
            cleared_metic_filter = [x for x in metric_filter if x in supported_metrics]
            if len(cleared_metic_filter) == 0:
                cleared_metic_filter = supported_metrics
        return cleared_metic_filter
        
    def plot_metrics(self, periods=['historical'], metric_type='evaluation', metric_filter=None):
        metric_filter = self.__get_supported_metrics(metric_filter)
        for metric in metric_filter:
            self.__plot_metrics(periods, metric_type, metric)

    def __plot_metrics(self, periods, metric_type, metric):
        
        title = self.__get_figure_title(metric_type, metric)
        ylabel = metric.upper()
        
        estimation_types = []
        if 'historical' in periods:
            estimation_types += ['_m','_h']
        if 'deferred' in periods:
            estimation_types += ['_d_m','_d_h']
        if 'future' in periods:
            estimation_types += ['__m_h','_f_m_h']

        data = self.__get_figure_data(metric_type, metric, estimation_types)
        
        labels = data[estimation_types[0]]['labels']
        x = np.arange(len(labels))  # the label locations

        plt.rcParams['figure.dpi'] = 100
        
        if len(estimation_types) == 2:
            width = 0.35
            offsets = [-0.5, 0.5]
            plt.rcParams['figure.figsize'] = [6, 6]
            plt.rcParams["font.size"] = "8"
        elif len(estimation_types) == 4:
            width = 0.20
            offsets = [-1.5, -0.5, 0.5, 1.5]
            plt.rcParams['figure.figsize'] = [8, 6]
            plt.rcParams["font.size"] = "7"
        elif len(estimation_types) == 6:
            width = 0.14
            offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
            plt.rcParams['figure.figsize'] = [10, 6]
            plt.rcParams["font.size"] = "7"
        else:
            raise Exception("Length of estimation_types is {} (instead of 2 or 4 or 6).".format(len(estimation_types)))

        fig, ax = plt.subplots()
        for i, offset in enumerate(offsets):
            rects = ax.bar(x + offset*width, data[estimation_types[i]]['values'],
                           width, label=data[estimation_types[i]]['description'])
            autolabel(ax, rects)
            assert data[estimation_types[0]]['labels'] == data[estimation_types[i]]['labels']

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.xticks(rotation = 45)

        fig.tight_layout()

        plt.show()
        
    def plot_confidence_comparison(self, period='historical', metric_filter=None):
        metric_filter = self.__get_supported_metrics(metric_filter)
        for metric in metric_filter:
            self.__plot_confidence_comparison(period, metric)

    def __plot_confidence_comparison(self, period, metric):

        plt.rcParams['figure.figsize'] = [12, 4]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"
        
        ylabel = metric.upper()
        metric_max = []
        metric_min = []

        def figure(ax, period, metric_type, metric, ylabel, title):
            
            if period == 'historical':
                estimation_types = ['_m','_h']
            elif period == 'deferred':
                estimation_types = ['_d_m','_d_h']
            elif period == 'future':
                estimation_types = ['__m_h','_f_m_h']

            data = self.__get_figure_data(metric_type, metric, estimation_types)

            labels = data[estimation_types[0]]['labels']
            x = np.arange(len(labels))

            metric_max.append(np.concatenate([data[estimation_types[0]]['values'],
                                              data[estimation_types[1]]['values']]).max())
            metric_min.append(np.concatenate([data[estimation_types[0]]['values'],
                                              data[estimation_types[1]]['values']]).min())
            
            width = 0.35
            offsets = [-0.5, 0.5]
            for i, offset in enumerate(offsets):
                rects = ax.bar(x + offset*width, data[estimation_types[i]]['values'],
                               width, label=data[estimation_types[i]]['description'])
                autolabel(ax, rects)
                assert data[estimation_types[0]]['labels'] == data[estimation_types[i]]['labels']
            
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

        fig, axs = plt.subplots(1,2)
        plt.tight_layout() 

        title = "{}: {} (all product groups)".format(self.model.model_name, ylabel)
        figure(axs[0], period, 'evaluation', metric, ylabel, title=title)
        
        title = "{}: {} (confident product groups, MAPE threshold={}".format(self.model.model_name,
                                                                             ylabel,
                                                                             self.confidence_threshold)
        figure(axs[1], period, 'confidence', metric, ylabel, title=title)
        
        y_lim_max = 0 if max(metric_max) < 0 else max(metric_max) * 1.2
        y_lim_min = 0 if min(metric_min) > 0 else min(metric_min) * 1.2

        axs[0].set_ylim(y_lim_min, y_lim_max)
        axs[1].set_ylim(y_lim_min, y_lim_max)
        
        plt.show()
        
    def plot_confidence_comparison_account(self, period='historical'):
        for account in self.account_test_filter:
            self.__plot_confidence_comparison_account(period, account)
            
    def __plot_confidence_comparison_account(self, period, account):

        plt.rcParams['figure.figsize'] = [5, 4]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        title = "{}: {} ({})".format(self.model.model_name, account, period)

        fig, ax = plt.subplots()
        plt.tight_layout()

        df_e = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
        df_c = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)

        columns_e, columns_c, labels = list(), list(), list()
        
        if period == 'historical':
            metric_filter = ['r2_m','wape_m','mape_m','bias_m','sfa_m']
        elif period == 'deferred':
            metric_filter = ['r2_d_m','wape_d_m','mape_d_m','bias_d_m','sfa_d_m']
        elif period == 'future':
            metric_filter = ['r2_f_m_h','wape_f_m_h','mape_f_m_h','bias_f_m_h','sfa_f_m_h']

        for column in df_e.columns:
            if account in column:
                for metric in metric_filter:
                    if column[-len(metric):] == metric:
                        columns_e.append(column)
                        labels.append(metric[:metric.find('_')])
        for column in df_c.columns:
            if account in column:
                for metric in metric_filter:
                    if column[-len(metric):] == metric:
                        columns_c.append(column)

        evaluation = df_e[columns_e].squeeze().values
        confidence = df_c[columns_c].squeeze().values
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        rects1 = ax.bar(x - width/2, evaluation, width, label='All products')
        rects2 = ax.bar(x + width/2, confidence, width,
                        label='Confident products, MAPE < ' + str(self.confidence_threshold))

        autolabel(ax, rects1)
        autolabel(ax, rects2)

        ylabel = self.__estimation_type_description(metric_filter[0][2:])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        plt.show()
        
    def __get_figure_data(self, metric_type, metric, estimation_types):
        
        if metric_type == 'evaluation':
            df = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
        elif metric_type == 'confidence':
            df = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)
        
        data = dict()
        for estimation_type in estimation_types:
            d = dict()
            columns = list()
            labels = list()
            for column in df.columns:
                if metric + estimation_type in column:
                    columns.append(column)
                    labels.append(column[:column.find(' /')])
            d['values'] = df[columns].squeeze().values
            d['labels'] = labels
            d['description'] = self.__estimation_type_description(estimation_type)
            data[estimation_type] = d
        
        return data

    def __estimation_type_description(self, estimation_type):
        description = {
            '_m': 'Model vs Fact (hist.)',
            '_h': 'Human vs Fact (hist.)',
            '_d_m': 'Model vs Fact (deferred)',
            '_d_h': 'Human vs Fact (deferred)',
            '__m_h': 'Model vs Human (hist.)',
            '_f_m_h': 'Model vs Human (future)',
        }
        return description.get(estimation_type, 'Description not found')
    
    def __set_figure_size(self):
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"
        
    def __get_figure_title(self, metric_type, metric):
        title = "{}: {} ({})".format(self.model.model_name,
                                   metric.upper(),
                                   'all product groups' if metric_type=='evaluation' else 'confident product groups')
        return title
        
    def overall_metrics_dataframe(self, metric_type):
        if metric_type not in ['evaluation','confidence']:
            metric_type = 'evaluation'
        # Select metrics
        if metric_type == 'evaluation':
            df = self.evaluation_overalls.copy()
        else:
            df = self.confidence_overalls.copy()
        df_list = []
        df = df.drop(['test','train','train_r2'], axis=1)
        metrics = ['r2','mape','sfa','bias','wape']
        for metric in metrics:
            columns = list()
            for column in df.columns:
                if metric in column:
                    columns.append(column)
            df_new = df[columns].T.reset_index()
            df_new['index'] = df_new['index'].apply(lambda x: x[len('Overall / ')+len(metric)+1:])
            df_new = df_new.set_index('index')
            df_new.columns = [metric]
            df_list.append(df_new)
        df_joined = pd.concat(df_list, axis=1)
        replace_dict = {
            'index': {
                'm': '1. Model vs Fact',
                'h': '2. Human vs Fact',
                'd_m': '3. Model vs Fact (deferred)',
                'd_h': '4. Human vs Fact (deferred)',
                '_m_h': '5. Model vs Human (hist.)',
                '_d_m_h': '6. Model vs Human (deferred)',
                'f_m_h': '7. Model vs Human (future)'}
        }
        df_joined.reset_index(inplace=True)
        df_joined = df_joined.replace(replace_dict)
        return df_joined.sort_values(['index'])

    def account_metrics_dataframe(self, metric_type, account):
        if metric_type not in ['evaluation','confidence']:
            metric_type = 'evaluation'
        # Select metrics
        if metric_type == 'evaluation':
            df = self.evaluation_accounts.copy()
        else:
            df = self.confidence_accounts.copy()
        df_list = []
        metrics = ['r2','mape','sfa','bias','wape']
        for metric in metrics:
            columns = list()
            for column in df.columns:
                if metric in column and account in column:
                    columns.append(column)
            df_new = df[columns].T.reset_index()
            df_new['index'] = df_new['index'].apply(lambda x: x[len(account)+len(metric)+4:])
            df_new = df_new.set_index('index')
            df_new.columns = [metric]
            df_list.append(df_new)
        df_joined = pd.concat(df_list, axis=1)
        replace_dict = {
            'index': {
                'm': '1. Model vs Fact',
                'h': '2. Human vs Fact',
                'd_m': '3. Model vs Fact (deferred)',
                'd_h': '4. Human vs Fact (deferred)',
                '_m_h': '5. Model vs Human (hist.)',
                '_d_m_h': '6. Model vs Human (deferred)',
                'f_m_h': '7. Model vs Human (future)'}
        }
        df_joined.reset_index(inplace=True)
        df_joined = df_joined.replace(replace_dict)
        return df_joined.sort_values(['index'])

    def save_all_metrics(self):
        """
        Save all metrics to a file.
        
        """
        report_name = os.path.join(self.model.output_dir, self.model.model_name + '_all_metrics.xlsx')
        with pd.ExcelWriter(report_name) as writer:
            df = pd.concat([self.overall_metrics_dataframe(metric_type='evaluation'),
                            self.overall_metrics_dataframe(metric_type='confidence')], axis=1)
            df.to_excel(writer, sheet_name='overalls')
            for account in self.model.account_test_filter:
                df = pd.concat([self.account_metrics_dataframe(metric_type='evaluation', account=account),
                                self.account_metrics_dataframe(metric_type='confidence', account=account)], axis=1)
                df.to_excel(writer, sheet_name=account)
                                   
        print("Model's metrics is saved to {}".format(report_name))
        
    def deferred_data_prediction(self):
        """ Save to a file the data for model and human accuracy comparison """
        # Report file name
        report_name = os.path.join(self.model.output_dir, self.model.model_name + '_deferred_data_prediction.xlsx')
        # Get data and make prediction
        df = self.model.deferred_data.copy()
        pred = self.model.model.predict(df)
        # Get required  fields from the data
        df = df[['promotion_name','account_banner','promotion_status','promotion_ext_id','original_product_dimension_26',
                 'prod_desc','yearweek','discount_perc','baseline_units_2',
                 'total_units_2','baseline_units','total_units','total_ef_qty']]
        # Calculate additional fields and cast to int
        df['lift_2'] = df['total_units_2'] / df['baseline_units_2']
        df['total_units_pred'] = pred
        df['lift_pred'] = df['total_units_pred'] / df['baseline_units_2']
        df['human_abs_error'] = np.abs(df['total_units_2'] - df['baseline_units_2'])
        df['model_abs_error'] = np.abs(df['total_units_pred'] - df['baseline_units_2'])
        df['wrong_promo'] = 0
        df['baseline_units_2'] = df['baseline_units_2'].astype(int)
        df['total_units_2'] = df['total_units_2'].astype(int)
        df['total_units_pred'] = df['total_units_pred'].astype(int)
        df['baseline_units'] = df['baseline_units'].astype(int)
        df['total_units'] = df['total_units'].astype(int)
        df['total_ef_qty'] = df['total_ef_qty'].astype(int)
        df['human_abs_error'] = df['human_abs_error'].astype(int)
        df['model_abs_error'] = df['model_abs_error'].astype(int)
        # Ordered fields
        df = df[['promotion_name','account_banner','promotion_status','promotion_ext_id','original_product_dimension_26',
                 'prod_desc','yearweek','discount_perc','baseline_units_2','total_units_2','lift_2','total_units_pred',
                 'lift_pred','baseline_units','total_units','total_ef_qty','human_abs_error','model_abs_error','wrong_promo']]
        # Saving
        with pd.ExcelWriter(report_name) as writer:
            df.to_excel(writer, sheet_name='report')
        
    def __tips_useful_methods(self):
        if not self.show_tips:
            return
        print("-- Reports --")
        print("analysis_metrics.confidence_criteria()")
        print("analysis_metrics.deferred_data_prediction()")
        print("-- Plotting --")
        print("analysis_metrics.plot_confidence_comparison(period='historical', metric_filter=None)")
        print("analysis_metrics.plot_confidence_comparison_account(period='historical')")
        print("analysis_metrics.plot_metrics(periods=['historical','deferred','future'],")
        print("                              metric_type='evaluation', metric_filter=['r2'])")
        print("-- Get metrics as dataframe --")
        print("analysis_metrics.overall_metrics_dataframe(metric_type)")
        print("analysis_metrics.account_metrics_dataframe(metric_type)")
        print("-- Save all metrics to file --")
        print("analysis_metrics.save_all_metrics()")
        print("-- Parameter's values --")
        print("period = 'historical' or 'deferred' or 'future'")
        print("metric_type = 'evaluation' or 'confidence'")
        print("metric_filter = ['r2','mape','sfa','bias','wape']")
