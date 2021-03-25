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
    
    def __init__(self, model, reevaluate=False, number_tests=10,
                 use_product_filter=True, filter_threshold=300, target_ratio_test=0.20):
        self.model = model
        self.full_model = model
        self.reevaluate = reevaluate
        self.number_tests = number_tests
        self.use_product_filter = use_product_filter
        self.filter_threshold = filter_threshold
        self.target_ratio_test = target_ratio_test

        self.__initialize_filter_data()
        self.__initialize_filter_future_data()
        self.__evaluate_model()
        
        self.overalls = self.overall_metrics()
        self.accounts = self.account_metrics()
        
    def __initialize_filter_data(self):
        """ Find a filter for data"""
        
        if self.model.filter_data is None and self.use_product_filter:
            print("Data filter initializing...")
            metrics = self.__mape_data_detailed()

            df = self.__get_account_detailed_metrics(['mape_m'], metrics)
            s = df.median()

            filts = dict()
            for i, item in enumerate(s[s > self.filter_threshold].items()):
                f1 = item[0].find("/")
                account = item[0][:f1 - 1]
                f2 = item[0].find("/",f1 + 1)
                product_25 = item[0][f1 + 1:f2 - 1]
                f3 = item[0].find("/",f2 + 1)
                product_26 = item[0][f2 + 2:f3 - 1]
                filt = dict()
                filt['account_banner'] = account
                filt['original_product_dimension_25'] = product_25
                filt['original_product_dimension_26'] = product_26
                filts[i] = filt

            self.filter_data = filts
        else:
            self.filter_data = self.model.filter_data

    def __initialize_filter_future_data(self):
        """ Find a filter for future data"""
        
        if self.model.filter_future_data is None and self.use_product_filter:
            print("Future data filter initializing...")
            metrics = self.__mape_future_data_detailed_test()

            df = self.__get_account_detailed_metrics(['mape_f_m_h'], metrics)
            s = df.median()

            filts = dict()
            for i, item in enumerate(s[s > self.filter_threshold].items()):
                f1 = item[0].find("/")
                account = item[0][:f1 - 1]
                f2 = item[0].find("/",f1 + 1)
                product_25 = item[0][f1 + 1:f2 - 1]
                f3 = item[0].find("/",f2 + 1)
                product_26 = item[0][f2 + 2:f3 - 1]
                filt = dict()
                filt['account_banner'] = account
                filt['original_product_dimension_25'] = product_25
                filt['original_product_dimension_26'] = product_26
                filts[i] = filt

            self.filter_future_data = filts
        else:
            self.filter_future_data = self.model.filter_future_data
            
    def __evaluate_model(self):
        self.metrics_data = self.model.metrics_data
        self.metrics_future_data = self.model.metrics_future_data
        if self.model.metrics_data is None or self.reevaluate:
            print("Evaluating the Model for historic data...")
            self.test_data()
        if self.model.metrics_future_data is None or self.reevaluate:
            print("Evaluating the Model for future data...")
            self.test_future_data()
            
    def __mape_data_detailed(self):
        metrics = list()
        for seed in range(self.number_tests):
            metrics.append(self.__mape_data_detailed_test_split(seed))
        return metrics

    def __mape_data_detailed_test_split(self, seed):

        if seed % 5 == 0:
            print("Test iteration {} of {}".format(seed + 1, self.number_tests))
        
        splitting = Splitting()
        self.train, self.test = splitting.split(seed, self.model.data, self.target_ratio_test)
        
        # Remove duped rows (possibly duplication has been provided)
        self.test = self.test.drop('cluster', axis=1).drop_duplicates()

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
                    metric_dict = self.get_metrics_by_account_product25_product26(['mape_m', 'mape_h'],
                                                                                account, product_25, product_26)
                    m_26['mape_m'] = metric_dict['mape_m']
                    m_26['mape_h'] = metric_dict['mape_h']
                    m_25[product_26] = m_26

                m_acc[product_25] = m_25

            m[account] = m_acc

        return m
    
    def __get_account_detailed_metrics(self, metric_filter, metrics):
        """Get metrics by accounts / product_25 / product_26"""
        lines = list()
        for item in metrics:
            line = dict()
            for key_acc in item.keys():
                if isinstance(item[key_acc], dict):
                    for key_25 in item[key_acc].keys():
                        if isinstance(item[key_acc][key_25], dict):
                            for key_26 in item[key_acc][key_25].keys():
                                if isinstance(item[key_acc][key_25][key_26], dict):
                                    for metric in item[key_acc][key_25][key_26].items():
                                        if metric[0] in metric_filter:
                                            line[key_acc + ' /' + key_25 + ' / ' + key_26 + ' / ' + metric[0]] = \
                                                item[key_acc][key_25][key_26][metric[0]]
            lines.append(line)

        df = pd.DataFrame(lines)
        df = df.drop_duplicates()
        return df

    def test_split(self, seed):

        if seed % 5 == 0:
            print("Test iteration {} of {}".format(seed + 1, self.number_tests))
        
        splitting = Splitting()
        self.train, self.test = splitting.split(seed, self.model.data, self.target_ratio_test)

        # Remove duped rows (possibly duplication has been provided)
        self.test = self.test.drop('cluster', axis=1).drop_duplicates()

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

            metric_dict = self.get_metrics_by_account(metric_filter, account)

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
            
            metric_dict = self.get_overall_metrics(metric_filter)
            
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
        
    def __mape_future_data_detailed_test(self):

        self.test = self.model.future_data.copy()
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
                        self.get_metrics_by_account_product25_product26(['mape_f_m_h'], account, product_25, product_26)
                    
                    m_26['mape_f_m_h'] = metric_dict['mape_f_m_h']
                    m_25[product_26] = m_26

                m_acc[product_25] = m_25

            m[account] = m_acc
            
        metrics = list()
        metrics.append(m)

        return metrics
    
    def test_future_data(self):

        self.test = self.model.future_data.copy()
        # Apply filter
        self.test = self.apply_product_filter(self.test, self.filter_future_data)
        self.test_x = self.test.loc[:, self.test.columns != self.model.future_target]
        self.test_y = self.test.loc[:, self.test.columns == self.model.future_target]
        self.pred_test_y = self.full_model.model.predict(self.test_x)

        m = dict()
        metric_filter = ['r2_f_m_h', 'wape_f_m_h', 'mape_f_m_h', 'bias_f_m_h', 'sfa_f_m_h']

        for account in self.test['account_banner'].unique():
            m_acc = dict()

            metric_dict = self.get_metrics_by_account(metric_filter, account)
            
            m_acc['r2_f_m_h'] = metric_dict['r2_f_m_h']
            m_acc['wape_f_m_h'] = metric_dict['wape_f_m_h']
            m_acc['mape_f_m_h'] = metric_dict['mape_f_m_h']
            m_acc['bias_f_m_h'] = metric_dict['bias_f_m_h']
            m_acc['sfa_f_m_h'] = metric_dict['sfa_f_m_h']

            for product_25 in self.test[self.test['account_banner']==account]['original_product_dimension_25'].unique():
                m_25 = dict()
                metric_dict = \
                    self.get_metrics_by_account_product25(metric_filter, account, product_25)
                
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
                        self.get_metrics_by_account_product25_product26(metric_filter, account, product_25, product_26)
                    
                    m_26['r2_f_m_h'] = metric_dict['r2_f_m_h']
                    m_26['wape_f_m_h'] = metric_dict['wape_f_m_h']
                    m_26['mape_f_m_h'] = metric_dict['mape_f_m_h']
                    m_26['bias_f_m_h'] = metric_dict['bias_f_m_h']
                    m_26['sfa_f_m_h'] = metric_dict['sfa_f_m_h']
                    m_25[product_26] = m_26

                m_acc[product_25] = m_25

            m[account] = m_acc

            # Overall metrics
            metric_dict = self.get_overall_metrics(metric_filter)
            
            m['r2_f_m_h'] = metric_dict['r2_f_m_h']
            m['wape_f_m_h'] = metric_dict['wape_f_m_h']
            m['mape_f_m_h'] = metric_dict['mape_f_m_h']
            m['bias_f_m_h'] = metric_dict['bias_f_m_h']
            m['sfa_f_m_h'] = metric_dict['sfa_f_m_h']

        self.metrics_future_data = m
        
    def apply_product_filter(self, data, filters):
        df = data.copy()
        if self.use_product_filter:
            for key in filters.keys():
                df = df[~((df['account_banner']==filters[key]['account_banner']) & \
                          (df['original_product_dimension_25']==filters[key]['original_product_dimension_25']) & \
                          (df['original_product_dimension_26']==filters[key]['original_product_dimension_26']))]
        return df

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

    def get_historic_overall_results(self):
        # Historic Data
        lines = list()
        for item in self.metrics_data:
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
        line = dict()
        line['Overall / r2_f_m_h'] = self.metrics_future_data['r2_f_m_h']
        line['Overall / wape_f_m_h'] = self.metrics_future_data['wape_f_m_h']
        line['Overall / mape_f_m_h'] = self.metrics_future_data['mape_f_m_h']
        line['Overall / bias_f_m_h'] = self.metrics_future_data['bias_f_m_h']
        line['Overall / sfa_f_m_h'] = self.metrics_future_data['sfa_f_m_h']
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
        data = data.drop_duplicates()
        data = data.quantile([0.5]).reset_index(drop=True)
        # Future test results
        future_data = self.get_future_overall_results()
        # Merge historic and future test results
        df = pd.concat([data, future_data], axis=1)
        return df
    
    def get_historic_account_results(self):
        # Historic Data
        lines = list()
        for item in self.metrics_data:
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
        line = dict()
        for key_acc in self.metrics_future_data.keys():
            if isinstance(self.metrics_future_data[key_acc], dict):
                line[key_acc + ' / r2_f_m_h'] = self.metrics_future_data[key_acc]['r2_f_m_h']
                line[key_acc + ' / wape_f_m_h'] = self.metrics_future_data[key_acc]['wape_f_m_h']
                line[key_acc + ' / mape_f_m_h'] = self.metrics_future_data[key_acc]['mape_f_m_h']
                line[key_acc + ' / bias_f_m_h'] = self.metrics_future_data[key_acc]['bias_f_m_h']
                line[key_acc + ' / sfa_f_m_h'] = self.metrics_future_data[key_acc]['sfa_f_m_h']
        lines.append(line)
        future_data = pd.DataFrame(lines)
        return future_data

    def account_metrics(self):
        """Get metrics by accounts"""
        # Historic test results
        data = self.get_historic_account_results()
        data = data.drop_duplicates()
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

    def calculate_metrics(self, metric_filter, filter):
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

    def get_overall_metrics(self, metric_filter):
        return self.calculate_metrics(metric_filter, filter=None)

    def get_metrics_by_account(self, metric_filter, account):
        filter = self.test['account_banner']==account
        return self.calculate_metrics(metric_filter, filter)

    def get_metrics_by_account_product25(self, metric_filter, account, product_25):
        filter = (self.test['account_banner']==account) & \
            (self.test['original_product_dimension_25']==product_25)
        return self.calculate_metrics(metric_filter, filter)

    def get_metrics_by_account_product25_product26(self, metric_filter, account, product_25, product_26):
        filter = (self.test['account_banner']==account) & \
            (self.test['original_product_dimension_25']==product_25) & \
            (self.test['original_product_dimension_26']==product_26)
        return self.calculate_metrics(metric_filter, filter)

    def plot_metrics(self, metric_filter=None):
        supported_metrics = list(['r2','wape','mape','bias','sfa'])
        if metric_filter is None:
            metric_filter = supported_metrics
        for metric in metric_filter:
            if metric in supported_metrics:
                self.plot_metric_by_clients(metric, metric.upper(), 'Median ' + metric.upper() + ' by clients and model/human')
