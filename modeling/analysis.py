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
    
    def __init__(self, model, number_tests=None, filter_threshold=None,
                 account_test_filter=None, target_ratio_test=0.20):
        """
        - If the model has not been yet analysed (model.evaluation_metrics is None), evaluation of MAPE metric starts
        with all data (historical and future). Parameter number_tests defines number of tests. For each
        test, data randomly splits. Evaluation is needed to define filter that is used to estimate
        credence criteria. Then the calculation of all metrics starts (accounting parameter filter_threshold).
        - To not apply product filter set filter_threshold to 0.
        
        """
        self.model = model
        self.full_model = model
        print("Analysing {} model".format(self.model.model_name))
        # If evaluation is needed
        if not self.model.evaluation_metrics:
            print("The model is not evaluated.")
            self.number_tests = number_tests or 10
        elif not number_tests is None and number_tests != len(self.model.test_splits):
            print("The model was evaluated with a different number of tests.")
            self.model.evaluation_metrics = None
            self.model.test_splits = None
            self.model.filter_threshold = None
            self.number_tests = number_tests
        else:
            print("The model is already evaluated.")
            self.number_tests = len(self.model.test_splits)
            self.recalculate = False
        # If metric recalculation is needed
        if not self.model.evaluation_metrics:
            # Evaluation is needed, thus, recalculation too
            self.filter_threshold = 300 if filter_threshold is None else filter_threshold
            self.recalculate = True
        elif not self.model.filter_threshold is None and filter_threshold is None:
            # The model is evaluated, the new threshold is not defined
            self.filter_threshold = self.model.filter_threshold
            self.recalculate = False
        elif filter_threshold == self.model.filter_threshold and not filter_threshold is None:
            # The model is evaluated, the new threshold is the same
            self.filter_threshold = filter_threshold
            self.recalculate = False
        else:
            self.filter_threshold = 300 if filter_threshold is None else filter_threshold
            self.recalculate = True
        self.account_test_filter = account_test_filter
        self.target_ratio_test = target_ratio_test
        self.test_splitting = Splitting(splits=self.model.test_splits, data=self.model.data,
                                        number_tests=self.number_tests, target_ratio_test=self.target_ratio_test)
        self.__get_unique_index_splits()
        self.__evaluate_model()
        self.__initialize_filter_data()
        self.__initialize_filter_future_data()
        self.__calculate_model()
        
        self.overalls = self.overall_metrics()
        self.accounts = self.account_metrics()
                
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
        print("unique_index_splits: ", self.unique_index_splits)
    
    def __evaluate_model(self):
        if self.model.evaluation_metrics is None:
            print("Evaluating the model with all historical data...")
            evaluation_metrics = self.__evaluation_metrics()
            print("Evaluating the model with all future data...")
            evaluation_future_metrics = self.__evaluation_future_metrics()
            # To be stored with the model
            self.evaluation_metrics = evaluation_metrics
            self.evaluation_future_metrics = evaluation_future_metrics
        else:
            self.evaluation_metrics = self.model.evaluation_metrics
            self.evaluation_future_metrics = self.model.evaluation_future_metrics
            
    def __initialize_filter_data(self):
        """ Find a filter for data"""
        if self.filter_threshold != 0:
            df = self.__parse_detailed_metrics(['mape_m'], self.evaluation_metrics)
            s = df.median()
            filts = dict()
            for i, item in enumerate(s[s > self.filter_threshold].items()):
                f1 = item[0].find("/")
                account = item[0][:f1 - 1]
                f2 = item[0].find("/",f1 + 1)
                product_25 = item[0][f1 + 2:f2 - 1]
                f3 = item[0].find("/",f2 + 1)
                product_26 = item[0][f2 + 2:f3 - 1]
                filt = dict()
                filt['account_banner'] = account
                filt['original_product_dimension_25'] = product_25
                filt['original_product_dimension_26'] = product_26
                filts[i] = filt
            self.filter_data = filts
        else:
            self.filter_data = None

    def __initialize_filter_future_data(self):
        """ Find a filter for future data"""
        if self.filter_threshold != 0:
            df = self.__parse_detailed_metrics(['mape_f_m_h'], self.evaluation_future_metrics)
            s = df.median()
            filts = dict()
            for i, item in enumerate(s[s > self.filter_threshold].items()):
                f1 = item[0].find("/")
                account = item[0][:f1 - 1]
                f2 = item[0].find("/",f1 + 1)
                product_25 = item[0][f1 + 2:f2 - 1]
                f3 = item[0].find("/",f2 + 1)
                product_26 = item[0][f2 + 2:f3 - 1]
                filt = dict()
                filt['account_banner'] = account
                filt['original_product_dimension_25'] = product_25
                filt['original_product_dimension_26'] = product_26
                filts[i] = filt
            self.filter_future_data = filts
        else:
            self.filter_future_data = None
            
    def __calculate_model(self):
        self.metrics_data = self.model.metrics_data
        self.metrics_future_data = self.model.metrics_future_data
        if self.model.metrics_data is None or self.recalculate:
            message = "Calulation of metrics for historical data"
            print(message + ". No product filter defined." if self.filter_threshold == 0 \
                  else message + " with 'MAPE' filter threshold {}.".format(self.filter_threshold))
            self.test_data()
        if self.model.metrics_future_data is None or self.recalculate:
            message = "Calulation of metrics for future data"
            print(message + ". No product filter defined." if self.filter_threshold == 0 \
                  else message + " with 'MAPE' filter threshold {}.".format(self.filter_threshold))
            self.test_future_data()
            
    def __evaluation_metrics(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.__evaluation_metrics_test_split(seed))
        return metrics

    def __evaluation_metrics_test_split(self, seed):

        if seed % 5 == 0:
            print("Test iteration {} of {}".format(seed + 1, self.number_tests))
        
        self.train, self.test = self.test_splitting.get_split(seed, self.model.data)
        
        # Remove duped rows (possibly duplication has been provided)
        self.test = self.test.drop('cluster', axis=1).drop_duplicates()

        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]
        
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

        for account in self.test['account_banner'].unique():
            m_acc = dict()

            for product_25 in self.test[self.test['account_banner']==account]['original_product_dimension_25'].unique():
                m_25 = dict()

                for product_26 in self.test[(self.test['account_banner']==account) & \
                                            (self.test['original_product_dimension_25']==product_25)] \
                                        ['original_product_dimension_26'].unique():
                    m_26 = dict()
                    metric_dict = self.__get_metrics_by_account_product25_product26(['mape_m','bias_m'],
                                                                                account, product_25, product_26)
                    m_26['mape_m'] = metric_dict['mape_m']
                    m_26['bias_m'] = metric_dict['bias_m']
                    m_25[product_26] = m_26

                m_acc[product_25] = m_25

            m[account] = m_acc

        return m
    
    def __parse_detailed_metrics(self, metric_filter, metrics):
        """Get metrics by accounts / product_25 / product_26"""
        lines = list()
        for i, item in enumerate(metrics):
            if len(metrics) > 1 and i not in self.unique_index_splits:
                # if metrics are related to historical and split not unique
                continue
            line = dict()
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    for key_25 in item[key_acc].keys():
                        if isinstance(item[key_acc][key_25], dict):
                            for key_26 in item[key_acc][key_25].keys():
                                if isinstance(item[key_acc][key_25][key_26], dict):
                                    for metric in item[key_acc][key_25][key_26].items():
                                        if metric[0] in metric_filter:
                                            line[key_acc + ' / ' + key_25 + ' / ' + key_26 + ' / ' + metric[0]] = \
                                                item[key_acc][key_25][key_26][metric[0]]
            lines.append(line)

        df = pd.DataFrame(lines)
        return df

    def test_split(self, seed):

        if seed % 5 == 0:
            print("Test iteration {} of {}".format(seed + 1, self.number_tests))
        
        self.train, self.test = self.test_splitting.get_split(seed, self.model.data)

        # Remove duped rows (possibly duplication has been provided)
        self.test = self.test.drop('cluster', axis=1).drop_duplicates()
        
        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]

        # Exclude products with large mape metric from test
        self.test = self.apply_product_filter(self.test, self.filter_data)
    
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
        
        for account in self.test['account_banner'].unique():
            m_acc = dict()

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

    def test_data(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.test_split(seed))
        self.metrics_data = metrics
        
    def __evaluation_future_metrics(self):

        self.test = self.model.future_data.copy()
        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]
        self.test_x = self.test.loc[:, self.test.columns != self.model.future_target]
        self.test_y = self.test.loc[:, self.test.columns == self.model.future_target]
        self.pred_test_y = self.full_model.model.predict(self.test_x)

        m = dict()

        for account in self.test['account_banner'].unique():
            m_acc = dict()

            for product_25 in self.test[self.test['account_banner']==account]['original_product_dimension_25'].unique():
                m_25 = dict()

                for product_26 in self.test[(self.test['account_banner']==account) & \
                                            (self.test['original_product_dimension_25']==product_25)] \
                                        ['original_product_dimension_26'].unique():
                    m_26 = dict()
                    metric_dict = \
                        self.__get_metrics_by_account_product25_product26(['mape_f_m_h','bias_f_m_h'],
                                                                          account, product_25, product_26)
                    
                    m_26['mape_f_m_h'] = metric_dict['mape_f_m_h']
                    m_26['bias_f_m_h'] = metric_dict['bias_f_m_h']
                    m_25[product_26] = m_26

                m_acc[product_25] = m_25

            m[account] = m_acc
            
        metrics = list()
        metrics.append(m)

        return metrics
    
    def test_future_data(self):

        self.test = self.model.future_data.copy()
        # Apply account test filter
        if not self.account_test_filter is None:
            self.test = self.test[self.test['account_banner'].isin(self.account_test_filter)]
        # Apply product test filter
        self.test = self.apply_product_filter(self.test, self.filter_future_data)
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
        self.metrics_future_data = metrics
        
    def apply_product_filter(self, data, filters):
        df = data.copy()
        if self.filter_threshold != 0:
            for key in filters.keys():
                df = df[~((df['account_banner']==filters[key]['account_banner']) & \
                          (df['original_product_dimension_25']==filters[key]['original_product_dimension_25']) & \
                          (df['original_product_dimension_26']==filters[key]['original_product_dimension_26']))]
        return df

    def get_historic_overall_results(self):
        # Historic Data
        lines = list()
        for i, item in enumerate(self.metrics_data):
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
    
    def get_future_overall_results(self):
        # Future data
        lines = list()
        for item in self.metrics_future_data:
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
    
    def overall_metrics(self):
        """Get overall metrics"""
        # Historic test results
        data = self.get_historic_overall_results()
        data = data.drop('seed', axis=1)
        data = data.quantile([0.5]).reset_index(drop=True)
        # Future test results
        future_data = self.get_future_overall_results()
        # Merge historic and future test results
        df = pd.concat([data, future_data], axis=1)
        return df
    
    def get_historic_account_results(self):
        # Historic Data
        lines = list()
        for i, item in enumerate(self.metrics_data):
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
        
    def get_future_account_results(self):
        # Future data
        lines = list()
        for item in self.metrics_future_data:
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

    def account_metrics(self):
        """Get metrics by accounts"""
        # Historic test results
        data = self.get_historic_account_results()
        data = data.quantile([0.5]).reset_index(drop=True)
        # Future test results
        future_data = self.get_future_account_results()
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
    
    
    def plot_metric_by_clients(self, metric_name, ylabel, title):
        
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = "8"

        df = pd.concat([self.overalls, self.accounts], axis=1)
        
        columns_m = list()
        columns_h = list()
        columns_m_h = list()
        columns_f_m_h = list()
        labels = list()
        for column in df.columns:
            if metric_name + '_m' in column:
                columns_m.append(column)
                labels.append(column[:-len(metric_name)-4])
            elif metric_name + '_h' in column:
                columns_h.append(column)
            elif metric_name + '__m' in column:
                columns_m_h.append(column)
            elif metric_name + '_f_m' in column:
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

    def plot_metrics(self, metric_filter=None):
        supported_metrics = list(['r2','wape','mape','bias','sfa'])
        if metric_filter is None:
            metric_filter = supported_metrics
        for metric in metric_filter:
            if metric in supported_metrics:
                self.plot_metric_by_clients(metric, metric.upper(), 'Median ' + metric.upper() + ' by clients and model/human')

    def get_credence_criteria(self):
        if self.filter_threshold == 0:
            print("Product filter is not defined.")
            return
        print("MAPE Threshold: {}".format(self.filter_threshold))
        print("Product groups")
        df = pd.DataFrame(self.filter_data).T.sort_values(
            ['account_banner','original_product_dimension_25','original_product_dimension_26'])
        print(df.to_string())
    
    def get_filter_future_data(self):
        if self.filter_threshold == 0:
            print("Product filter is not defined.")
            return
        print("Future test filter for threshold {}:".format(self.filter_threshold))
        df = pd.DataFrame(self.filter_future_data).T.sort_values(
            ['account_banner','original_product_dimension_25','original_product_dimension_26'])
        print(df.to_string())

    def get_product_filter(self, filter='historic'):
        if filter == 'historic':
            filters = self.filter_data
            print("historic filter:")
        else:
            filters = self.filter_future_data
            print("future filter:")
        for key in filters.keys():
            print("{} / {} / {}".format(filters[key]['account_banner'],
                                        filters[key]['original_product_dimension_25'],
                                        filters[key]['original_product_dimension_26']))