import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from vf_portalytics.multi_model import MultiModel
import matplotlib.pyplot as plt

from splitting import Splitting
from utils import suppress_stdout


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
                                        number_tests=self.number_tests, target_ratio_test=self.target_ratio_test)
        self.__get_unique_index_splits()
        self.__evaluate_model()
        self.__initialize_confidence_filter()
        self.__initialize_confidence_future_filter()
        self.__calculate_model()
        
        self.evaluation_overalls = self.__overall_metrics('evaluation')
        self.evaluation_accounts = self.__account_metrics('evaluation')
        self.confidence_overalls = self.__overall_metrics('confidence')
        self.confidence_accounts = self.__account_metrics('confidence')
        
        self.__update_model()
        self.__tips_useful_methods()
        
    def __update_model(self):
        self.model.test_splits = self.test_splitting.splits
        self.model.evaluation_metrics = self.evaluation_metrics
        self.model.evaluation_future_metrics = self.evaluation_future_metrics
        self.model.account_test_filter = self.account_test_filter
        self.model.confidence_threshold = self.confidence_threshold
        self.model.confidence_metrics = self.confidence_metrics
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
            print("Evaluating the model with all historical data...")
            self.__evaluation_metrics()
            print("Evaluating the model with all future data...")
            self.__evaluation_future_metrics()
        else:
            self.evaluation_metrics = self.model.evaluation_metrics
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

    def __initialize_confidence_future_filter(self):
        """ Find a filter for future data"""
        if self.confidence_threshold != 0:
            df = self.__parse_detailed_metrics(['mape_f_m_h'], self.evaluation_future_metrics)
            df = (df.groupby(['account_banner','original_product_dimension_25','original_product_dimension_26'])
                              .median().reset_index())
            filts = dict()
            for i, row in enumerate(df[df['mape_f_m_h'] > self.confidence_threshold].itertuples()):
                filt = dict()
                filt['account_banner'] = row[1]
                filt['original_product_dimension_25'] = row[2]
                filt['original_product_dimension_26'] = row[3]
                filts[i] = filt
            self.filter_future_data = filts
        else:
            self.filter_future_data = None
            
    def __calculate_model(self):
        self.confidence_metrics = self.model.confidence_metrics
        self.confidence_future_metrics = self.model.confidence_future_metrics
        if self.model.confidence_metrics is None or self.recalculate:
            message = "Calulation of metrics for historical data"
            print(message + ". No confidence threshold defined." if self.confidence_threshold == 0 \
                  else message + " with 'MAPE' confidence threshold {}.".format(self.confidence_threshold))
            self.__confidence_metrics()
        if self.model.confidence_future_metrics is None or self.recalculate:
            message = "Calulation of metrics for future data"
            print(message + ". No confidence threshold defined." if self.confidence_threshold == 0 \
                  else message + " with 'MAPE' confidence threshold {}.".format(self.confidence_threshold))
            self.__confidence_future_metrics()
            
    def __evaluation_metrics(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.__metrics_split(seed, apply_confidence_filter=False))
        self.evaluation_metrics = metrics

    def __evaluation_future_metrics(self):
        metrics = self.__future_metrics_split(apply_confidence_filter=False)
        self.evaluation_future_metrics = metrics
    
    def __confidence_metrics(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.__metrics_split(seed, apply_confidence_filter=True))
        self.confidence_metrics = metrics

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

        # Remove duped rows (possibly duplication has been provided)
        self.test = self.test.drop('cluster', axis=1).drop_duplicates()
        
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
    #         test_x['baseline_units_original'] = test_x['baseline_units']
    #         test_x['baseline_units'] = test_x['baseline_units_2']
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

    def __future_metrics_split(self, apply_confidence_filter):

        self.test = self.model.future_data.copy()
        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]
        # Apply product test filter
        if apply_confidence_filter:
            self.test = self.__apply_confidence_filter(self.test, self.filter_future_data)
        self.test_x = self.test.loc[:, self.test.columns != self.model.future_target]
        self.test_y = self.test.loc[:, self.test.columns == self.model.future_target]
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

            for product_25 in self.test[self.test['account_banner']==account]['original_product_dimension_25'].unique():
                m_25 = dict()
                metric_dict = \
                    self.__get_metrics_by_account_product25(metric_filter, account, product_25)
                
                m_25['r2_f_m_h'] = metric_dict['r2_f_m_h']
                m_25['wape_f_m_h'] = metric_dict['wape_f_m_h']
                m_25['mape_f_m_h'] = metric_dict['mape_f_m_h']
                m_25['bias_f_m_h'] = metric_dict['bias_f_m_h']
                m_25['sfa_f_m_h'] = metric_dict['sfa_f_m_h']

                for product_26 in self.test[(self.test['account_banner']==account) & \
                                            (self.test['original_product_dimension_25']==product_25)] \
                                        ['original_product_dimension_26'].unique():
                    m_26 = dict()
                    metric_dict = \
                        self.__get_metrics_by_account_product25_product26(metric_filter, account, product_25, product_26)
                    
                    m_26['r2_f_m_h'] = metric_dict['r2_f_m_h']
                    m_26['wape_f_m_h'] = metric_dict['wape_f_m_h']
                    m_26['mape_f_m_h'] = metric_dict['mape_f_m_h']
                    m_26['bias_f_m_h'] = metric_dict['bias_f_m_h']
                    m_26['sfa_f_m_h'] = metric_dict['sfa_f_m_h']
                    m_25[product_26] = m_26

                m_acc[product_25] = m_25

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
    
    def __future_overall_metrics(self, future_metrics):
        # Future data
        lines = list()
        for item in future_metrics:
            line = dict()
            line['Overall / r2_f_m_h'] = item['r2_f_m_h']
            line['Overall / wape_f_m_h'] = item['wape_f_m_h']
            line['Overall / mape_f_m_h'] = item['mape_f_m_h']
            line['Overall / bias_f_m_h'] = item['bias_f_m_h']
            line['Overall / sfa_f_m_h'] = item['sfa_f_m_h']
            lines.append(line)
        future_data = pd.DataFrame(lines)
        future_data = future_data[['Overall / r2_f_m_h','Overall / wape_f_m_h','Overall / mape_f_m_h',
                                   'Overall / bias_f_m_h','Overall / sfa_f_m_h']]
        return future_data
    
    def __overall_metrics(self, metric_type):
        """Get overall metrics"""
        if metric_type not in ['evaluation','confidence']:
            metric_type = 'evaluation'
        # Select metrics
        if metric_type == 'evaluation':
            metrics = self.evaluation_metrics
            future_metrics = self.evaluation_future_metrics
        else:
            metrics = self.confidence_metrics
            future_metrics = self.confidence_future_metrics
        # Historic test results            
        data = self.__historic_overall_metrics(metrics)
        data = data.drop('seed', axis=1)
        data = data.quantile([0.5]).reset_index(drop=True)
        # Future test results
        future_data = self.__future_overall_metrics(future_metrics)
        # Merge historic and future test results
        df = pd.concat([data, future_data], axis=1)
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
        
    def __future_account_metrics(self, future_metrics):
        # Future data
        lines = list()
        for item in future_metrics:
            line = dict()
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    line[key_acc + ' / r2_f_m_h'] = item[key_acc]['r2_f_m_h']
                    line[key_acc + ' / wape_f_m_h'] = item[key_acc]['wape_f_m_h']
                    line[key_acc + ' / mape_f_m_h'] = item[key_acc]['mape_f_m_h']
                    line[key_acc + ' / bias_f_m_h'] = item[key_acc]['bias_f_m_h']
                    line[key_acc + ' / sfa_f_m_h'] = item[key_acc]['sfa_f_m_h']
            lines.append(line)
        future_data = pd.DataFrame(lines)
        return future_data

    def __account_metrics(self, metric_type):
        """Get metrics by accounts"""
        if metric_type not in ['evaluation','confidence']:
            metric_type = 'evaluation'
        # Select metrics
        if metric_type == 'evaluation':
            metrics = self.evaluation_metrics
            future_metrics = self.evaluation_future_metrics
        else:
            metrics = self.confidence_metrics
            future_metrics = self.confidence_future_metrics
        # Historic test results
        data = self.__historic_account_metrics(metrics)
        data = data.quantile([0.5]).reset_index(drop=True)
        # Future test results
        future_data = self.__future_account_metrics(future_metrics)
        # Merge historic and future data
        df = pd.concat([data, future_data], axis=1)
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
    
    def get_filter_future_data(self):
        if self.confidence_threshold == 0:
            print("Product filter is not defined.")
            return
        print("Future test filter for threshold {}:".format(self.confidence_threshold))
        df = pd.DataFrame(self.filter_future_data).T.sort_values(
            ['account_banner','original_product_dimension_25','original_product_dimension_26'])
        print(df.to_string())

    def get_product_filter(self, filter='historic'):
        if filter == 'historic':
            filters = self.confidence_filter
            print("historic filter:")
        else:
            filters = self.filter_future_data
            print("future filter:")
        for key in filters.keys():
            print("{} / {} / {}".format(filters[key]['account_banner'],
                                        filters[key]['original_product_dimension_25'],
                                        filters[key]['original_product_dimension_26']))
            
    def __get_supported_metrics(self, metric_filter):
        supported_metrics = list(['r2','wape','mape','bias','sfa'])
        if metric_filter is None or not isinstance(metric_filter, list):
            cleared_metic_filter = supported_metrics
        else:
            cleared_metic_filter = [x for x in metric_filter if x in supported_metrics]
            if len(cleared_metic_filter) == 0:
                cleared_metic_filter = supported_metrics
        return cleared_metic_filter
            
    def plot_confidence_comparison(self, metric_filter=None):
        metric_filter = self.__get_supported_metrics(metric_filter)
        for metric in metric_filter:
            self.__plot_confidence_comparison(metric)

    def __plot_confidence_comparison(self, metric):

        plt.rcParams['figure.figsize'] = [12, 4]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"
        
        ylabel = metric.upper()

        def autolabel(ax, rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        def figure(ax, metric_type, metric, ylabel, title):
            if metric_type == 'evaluation':
                df = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
            elif metric_type == 'confidence':
                df = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)

            columns_m, columns_h, labels = list(), list(), list()
            for column in df.columns:
                if metric + '_m' in column:
                    columns_m.append(column)
                    labels.append(column[:-len(metric)-4])
                elif metric + '_h' in column:
                    columns_h.append(column)

            model = df[columns_m].squeeze().values
            human = df[columns_h].squeeze().values
            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars
            rects1 = ax.bar(x - width/2, model, width, label='Model vs Fact')
            rects2 = ax.bar(x + width/2, human, width, label='Human vs Fact')

            autolabel(ax, rects1)
            autolabel(ax, rects2)

            # Add some text for labels, title and custom x-axis tick labels, etc.
#             ax.set_ylim(0, 1)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

        fig, axs = plt.subplots(1,2)
        plt.tight_layout() 

        title = "Median {} (all product groups)".format(ylabel)
        figure(axs[0], 'evaluation', metric, ylabel, title=title)
        
        title = "Median {} (confident product groups, MAPE threshold={}".format(ylabel, self.confidence_threshold)
        figure(axs[1], 'confidence', metric, ylabel, title=title)

        plt.show()
        
    def plot_performance_metrics(self, metric_type='evaluation', metric_filter=None):
        metric_filter = self.__get_supported_metrics(metric_filter)
        for metric in metric_filter:
            self.__plot_performance_metric(metric_type, metric)

    def __plot_performance_metric(self, metric_type, metric):
        
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        ylabel = metric.upper()
        
        if metric_type == 'evaluation':
            df = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
            title = 'Median ' + ylabel + ' (all product groups)'
        elif metric_type == 'confidence':
            df = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)
            title = 'Median ' + ylabel + ' (confident product groups)'
        
        columns_m = list()
        columns_h = list()
        columns_m_h = list()
        columns_f_m_h = list()
        labels = list()
        for column in df.columns:
            if metric + '_m' in column:
                columns_m.append(column)
                labels.append(column[:-len(metric)-4])
            elif metric + '_h' in column:
                columns_h.append(column)
            elif metric + '__m' in column:
                columns_m_h.append(column)
            elif metric + '_f_m' in column:
                columns_f_m_h.append(column)

        df_m = df[columns_m]
        df_h = df[columns_h]
        df_m_h = df[columns_m_h]
        df_f_m_h = df[columns_f_m_h]

        model = df_m.squeeze().values
        human = df_h.squeeze().values
        model_human = df_m_h.squeeze().values
        future_model_human = df_f_m_h.squeeze().values
        
        x = np.arange(len(labels))  # the label locations
        width = 0.22  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width - width/2, model, width, label='Model vs Fact')
        rects2 = ax.bar(x - width/2, human, width, label='Human vs Fact')
        rects3 = ax.bar(x + width/2, model_human, width, label='Model vs Human (Historic)')
        rects4 = ax.bar(x + width + width/2, future_model_human, width, label='Model vs Human (Future)')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.xticks(rotation = 45)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        fig.tight_layout()

        plt.show()
        
    def plot_confidence_comparison_account(self):
        for account in self.account_test_filter:
            self.__plot_confidence_comparison_account(account)
            
    def __plot_confidence_comparison_account(self, account):

        plt.rcParams['figure.figsize'] = [5, 4]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        title = account

        def autolabel(ax, rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        fig, ax = plt.subplots()
        plt.tight_layout()

        df_e = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
        df_c = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)

        columns_e, columns_c, labels = list(), list(), list()
        metric_filter = ['r2_m','wape_m','mape_m','bias_m','sfa_m']
        for column in df_e.columns:
            if account in column:
                for metric in metric_filter:
                    if column[-len(metric):] == metric:
                        columns_e.append(column)
                        labels.append(metric)
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

        # Add some text for labels, title and custom x-axis tick labels, etc.
    #     ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        plt.show()
        
        
    def plot_performance_metrics_historic(self, metric_type='evaluation', metric_filter=None):
        metric_filter = self.__get_supported_metrics(metric_filter)
        for metric in metric_filter:
            self.__plot_performance_metrics_historic(metric_type, metric)

    def __plot_performance_metrics_historic(self, metric_type, metric):
        
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        ylabel = metric.upper()
        
        if metric_type == 'evaluation':
            df = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
            title = 'Median ' + ylabel + ' (all product groups)'
        elif metric_type == 'confidence':
            df = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)
            title = 'Median ' + ylabel + ' (confident product groups)'
        
        columns_m = list()
        columns_h = list()
        labels = list()
        for column in df.columns:
            if metric + '_m' in column:
                columns_m.append(column)
                labels.append(column[:-len(metric)-4])
            elif metric + '_h' in column:
                columns_h.append(column)

        df_m = df[columns_m]
        df_h = df[columns_h]

        model = df_m.squeeze().values
        human = df_h.squeeze().values
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, model, width, label='Model vs Fact')
        rects2 = ax.bar(x + width/2, human, width, label='Human vs Fact')

        # Add some text for labels, title and custom x-axis tick labels, etc.
#         ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.xticks(rotation = 45)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.show()

        
    def plot_performance_metrics_future(self, metric_type='evaluation', metric_filter=None):
        metric_filter = self.__get_supported_metrics(metric_filter)
        for metric in metric_filter:
            self.__plot_performance_metrics_future(metric_type, metric)

    def __plot_performance_metrics_future(self, metric_type, metric):
        
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        ylabel = metric.upper()
        
        if metric_type == 'evaluation':
            df = pd.concat([self.evaluation_overalls, self.evaluation_accounts], axis=1)
            title = 'Median ' + ylabel + ' (all product groups)'
        elif metric_type == 'confidence':
            df = pd.concat([self.confidence_overalls, self.confidence_accounts], axis=1)
            title = 'Median ' + ylabel + ' (confident product groups)'
        
        columns_m_h = list()
        columns_f_m_h = list()
        labels = list()
        for column in df.columns:
            if metric + '__m' in column:
                columns_m_h.append(column)
                labels.append(column[:-len(metric)-4])
            elif metric + '_f_m' in column:
                columns_f_m_h.append(column)

        df_m_h = df[columns_m_h]
        df_f_m_h = df[columns_f_m_h]

        model_human = df_m_h.squeeze().values
        future_model_human = df_f_m_h.squeeze().values
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, model_human, width, label='Model vs Human (Historic)')
        rects2 = ax.bar(x + width/2, future_model_human, width, label='Model vs Human (Future)')

        # Add some text for labels, title and custom x-axis tick labels, etc.
#         ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.xticks(rotation = 45)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.show()
        
    def __tips_useful_methods(self):
        if not self.show_tips:
            return
        print("Useful methods:")
        print("analysis_metrics.confidence_criteria()")
        print("analysis_metrics.plot_confidence_comparison(metric_filter=None)")
        print("analysis_metrics.plot_confidence_comparison_account()")
        print("analysis_metrics.plot_performance_metrics(metric_type='evaluation', metric_filter=None)")
        print("analysis_metrics.plot_performance_metrics_historic(metric_type='evaluation', metric_filter=None)")
        print("analysis_metrics.plot_performance_metrics_future(metric_type='evaluation', metric_filter=None)")