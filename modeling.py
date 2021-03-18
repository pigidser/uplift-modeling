import os
import numpy as np
import pandas as pd
import json

from vf_portalytics.multi_model import MultiModel
from vf_portalytics.model import PredictionModel

from hyperparams import Hyperparameters
from analysis import AnalysisMetrics


class Model(object):

    def __init__(self, model_name):
        """
        
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.model_name = model_name
        self.metrics_data = None
        self.metrics_future_data = None
        self.filter_data = None
        self.filter_future_data = None
        self.model_initialized = False
        
    def get_info(self):
        print("--- Model Information ---")
        print("Name: {}".format(self.model_name))
        if not self.model_initialized:
            print("Not initialized")
            return
        print("feature_filename: {}".format(self.feature_filename))
        print("output_dir: {}".format(self.output_dir))
        print("data_filename : {}".format(self.data_filename))
        print("filter_filename : {}".format(self.filter_filename))
        print("target : {}".format(self.target))
        print("future_data_filename : {}".format(self.future_data_filename))
        print("future_target : {}".format(self.future_target))
        print("filter_data : {}".format('Not defined' if self.filter_data is None else 'Defined'))
        print("filter_future_data : {}".format('Not defined' if self.filter_future_data is None else 'Defined'))
        print("metrics_data : {}".format('Not defined' if self.metrics_data is None else 'Defined'))
        print("metrics_future_data : {}".format('Not defined' if self.metrics_future_data is None else 'Defined'))
        print("account_filter : {}".format(self.account_filter))
        print("-------------------------")
        
    def update_info(self, analysis):
        self.filter_data = analysis.filter_data
        self.filter_future_data = analysis.filter_future_data
        self.metrics_data = analysis.metrics_data
        self.metrics_future_data = analysis.metrics_future_data
        
    def create(self, params, feature_filename, target, cat_feature,
                     output_dir, data_filename, filter_filename, future_data_filename, future_target):
        """
        Define the next artifacts related to input
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.params = params
        self.feature_filename = feature_filename
        self.target = target
        self.cat_feature = cat_feature
        self.output_dir = output_dir
        self.data_filename = data_filename
        self.filter_filename = filter_filename
        self.future_data_filename = future_data_filename
        self.future_target = future_target
        self.filter_data = None
        self.filter_future_data = None
        self.metrics = None
        
        self.__load_dataset()
        self.__load_future_data()
        self.__load_features()
        self.__set_clusters()
        self.__set_params()

        self.model_initialized = True
        self.apply_account_filter()
    
    def check_status(self):
        if not self.model_initialized:
            raise Exception("Please create or load a model first")
        return True
    
    def __find_account_filter(self):
        self.check_status()
        self.account_filter = self.data['account_banner'].unique()
        return self.account_filter

    def apply_account_filter(self, new_filter=None):
        self.__find_account_filter()
        if not new_filter is None:
            self.account_filter = new_filter
        self.data = self.data[self.data['account_banner'].isin(self.account_filter)]
        self.future_data = self.future_data[self.future_data['account_banner'].isin(self.account_filter)]

    def get_account_filter(self):
        print(self.__find_account_filter())
        
    def __load_dataset(self):
        """
        
        """
        # Load dataset
        if not os.path.isfile(self.data_filename):
            raise IOError, "Please specify a correct file with train dataset. File not found: " + self.data_filename
        if not os.path.isfile(self.filter_filename):
            raise IOError, "Please specify a correct file with a filter for train dataset. " + \
                "File not found: " + self.data_filename
        df = pd.read_msgpack(self.data_filename)
        # Get filter for dataset
        with open(self.filter_filename, 'r') as file:
            remaining_index_dict = json.load(file)
        remaining_index = remaining_index_dict['remaining_index']
        # Apply data filter
        mask = df.index.isin(remaining_index)
        df = df[mask]
        # Replace inf with nans
        df = df.replace([np.inf, -np.inf], np.nan)
        # Fill numeric values with 0.0
        for column in df.select_dtypes(include=[np.int64, np.float64, np.float32]):
            df[column] = df[column].fillna(0.0)
        self.data = df
        
    def __load_future_data(self):
        if not os.path.isfile(self.future_data_filename):
            raise IOError, "Please specify a correct file with train dataset. File not found: " \
                + self.future_data_filename
        df = pd.read_msgpack(self.future_data_filename)
        
        # Filter by status
        mask = df['promotion_status'] >= 110
        print("Filtered due to status of promotion".format(len(df[mask])))
        df = df[~mask]

        # Remove products that do not exist in train dataset
        mask = ~df['original_pid'].isin(self.data['original_pid'])
        print("Filtered due to products not exist in train dataset".format(len(df[mask])))
        df = df[~mask]

        # Remove exclusions
        mask = df['exclude_yn'].isin([1, True])
        print("Filtered due to excluded flag".format(len(df[mask])))
        df = df[~mask]

        # Baseline
        mask = df['baseline_units_2'] == 0.0
        print("Filtered due to zero planned baseline".format(len(df[mask])))
        df = df[~mask]

        # Baseline
        mask = df['total_units_2'] == 0.0
        print("Filtered due to zero planned in-store total volume".format(len(df[mask])))
        df = df[~mask]

        mask = df['total_units_2'] <= df['baseline_units_2']
        print("Filtered due to planned in-store total volume lower then baseline".format(len(df[mask])))
        df = df[~mask]
        
        print("Future dataset length: ".format(len(df)))
        
        self.future_data = df
        
    def __load_features(self):
        if not os.path.isfile(self.feature_filename):
            raise IOError, "Please specify a correct file with the information about features. " + \
                "File not found: " + self.feature_filename
        # Feature retrieval
        with open(self.feature_filename, 'r') as file:
            feature_info_dict = json.load(file)
        # Take only this information. Do not take the remaining_feat
        self.features = feature_info_dict['remaining_feat'] 
        self.ordinal_features = feature_info_dict['ordinal_features'] 
        self.nominal_features = feature_info_dict['nominal_features'] 

    def __set_clusters(self):
        """
        We have the option to create a different model for each category
        (categories are based in a specific feature) or having one model
        
        """
        if self.cat_feature == 'vf_category' or self.cat_feature == None:
            # vf_category means that no category defined
            self.data[self.cat_feature] = 0.0

        self.clusters = self.data[self.cat_feature].unique()
        self.selected_features = {}
        for cluster in self.clusters:
            self.selected_features[cluster] = self.features

    def set_features(self, features):
        self.check_status()
        if not isinstance(features, list):
            raise Exception("The 'Features' variable type should be a list")
        self.features = features
        self.__set_clusters()
        
    def __set_params(self):
        if self.params == None or not isinstance(self.params, dict):
            hyper = Hyperparameters(self)
            hyper.find_params()
            self.params = hyper.params

    def train(self):
        self.check_status()
        train_x = self.data #[self.features]
        train_y = self.data[self.target]
        self.model = MultiModel(group_col=self.cat_feature,
                                clusters=self.clusters,
                                params=self.params,
                                selected_features=self.selected_features,
                                nominals=self.nominal_features,
                                ordinals=self.ordinal_features)
        self.model.fit(train_x, train_y)
        print("The model is trained")
        
#     def evaluate(self):
#         if not isinstance(self.model, MultiModel):
#             raise Exception("Train the model first")
#         analysis_metrics = AnalysisMetrics(self)
#         self.metrics = analysis_metrics.metrics
#         print("The model is evaluated")
    
    def __extended_save(self):
        # Additional information storing
        meta_path = os.path.join(os.getcwd(), self.model_name + '_ext.meta')
        meta_data = {
            'feature_filename': self.feature_filename,
            'output_dir': self.output_dir,
            'data_filename': self.data_filename,
            'filter_filename': self.filter_filename,
            'target': self.target,
            'future_data_filename': self.future_data_filename,
            'future_target': self.future_target,
            'metrics_data': self.metrics_data,
            'metrics_future_data': self.metrics_future_data,
            'account_filter': list(self.account_filter),
            'filter_data': self.filter_data,
            'filter_future_data': self.filter_future_data,
        }
        with open(meta_path, 'w') as meta_file:
            json.dump(meta_data, meta_file)
            
    def __standard_save(self):
        # VF related saving
        prediction_model = PredictionModel(self.model_name, path=self.output_dir,
                                           one_hot_encode=False)
        prediction_model.model = self.model
        # save feature names (no strictly since all the preprocessing is made being made in the pipeline)
        prediction_model.features = {key: [] for key in self.features}
        prediction_model.target = {self.target: []}

        prediction_model.ordered_column_list = sorted(prediction_model.features.keys())
        prediction_model.save()
        
    def save(self, model_name=None):
        """Save a model and artifacts to files"""
        if not isinstance(self.model, MultiModel):
            raise Exception("Train the model first")
        if not model_name is None:
            self.model_name = model_name
        self.__extended_save()
        self.__standard_save()
        print("The model is saved")
        
    def __extended_load(self):
        meta_path = os.path.join(os.getcwd(), self.model_name + '_ext.meta')
        if not os.path.isfile(meta_path):
            raise IOError, "Please specify a correct file with model's meta data. File not found: " \
                + meta_path
        else:
            with open(meta_path, 'r') as meta_file:
                meta_data = json.load(meta_file)
                self.feature_filename=meta_data.get('feature_filename')
                self.output_dir=meta_data.get('output_dir')
                self.data_filename=meta_data.get('data_filename')
                self.filter_filename=meta_data.get('filter_filename')
                self.target=meta_data.get('target')
                self.future_data_filename=meta_data.get('future_data_filename')
                self.future_target=meta_data.get('future_target')
                self.metrics_data = meta_data.get('metrics_data')
                self.metrics_future_data = meta_data.get('metrics_future_data')
                self.account_filter = meta_data.get('account_filter')
                self.filter_data = meta_data.get('filter_data')
                self.filter_future_data = meta_data.get('filter_future_data')

    def __standard_load(self):
        prediction_model = PredictionModel(self.model_name, path=self.output_dir, one_hot_encode=False)
        self.model = prediction_model.model
        
    def load(self):
        """Load a model and artifacts from files"""
        self.__extended_load()
        self.__standard_load()
        self.__load_dataset()
        self.__load_future_data()
        for key in self.model.selected_features:
            self.features = self.model.selected_features[key]
            break
        self.ordinal_features = self.model.ordinals 
        self.nominal_features = self.model.nominals
        self.cat_feature = self.model.group_col
        self.__set_clusters()
        self.params = self.model.params
        
        self.model_initialized = True
        self.apply_account_filter(self.account_filter)
        print("The model is loaded")