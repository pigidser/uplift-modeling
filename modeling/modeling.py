import os
import numpy as np
import pandas as pd
import json

from vf_portalytics.multi_model import MultiModel
from vf_portalytics.model import PredictionModel

from hyperparams import Hyperparameters
from analysis import AnalysisMetrics
from transforms import ClusterTransform

from sklearn.pipeline import Pipeline

class Model(object):
    
    """
    # Declare a model
    model = Model(model_name='model1')
    
    # Creating the model from scratch
    ---------------------------------
    
    # Use existing params or None. If params defined, max_evals not used.
    params = {0.0: {'colsample_bytree': 0.5,
      'gamma': 0.75,
      'learning_rate': 0.04,
      'max_depth': 5,
      'min_child_weight': 5.0,
      'n_estimators': 142,
      'reg_alpha': 0.9,
      'reg_lambda': 1.4500000000000002,
      'subsample': 0.55,
      'transformer_nominal': 'JamesSteinEncoder',
      'transformer_ordinal': 'OrdinalEncoder',
      'under_predict_weight': 2.0}}

    # Use feature set from feature_filename file or define another set
    features = [
        u'original_pid',
        u'account_id',
        u'original_product_dimension_25',
        u'original_product_dimension_26',
        u'week_agg_8',
        u'baseline_units',
        u'consumer_length',
        u'promotion_type',
        u'discount_perc_cohort',
        u'promoted_niv',
        u'previous_promotion_week_distance',
        u'total_nr_products'
    ]

    model.create(
        params=params,
        max_evals=None,
        target_ratio_val=None,
        feature_filename='./outputs/im_feature_info_dict_mars_ru_20210212.txt',
        features=features,
        target='total_units',
        cat_feature=None,
        output_dir='outputs',
        data_filename='../data/20210212_mars_ru_prod_trainset.msgpack',
        filter_filename='./outputs/im_data_retrieval-v6-20210212.txt',
        account_filter=['5Pyaterochka (X5)','Lenta','Dixy','Okey Group','Magnit'],
        future_data_filename='../data/20210212_mars_ru_prod_futureset.msgpack',
        future_target='total_units_2',
        duplication_map=None)

    model.train()

    """

    def __init__(self, model_name):
        """
        
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.model_name = model_name
        self.test_splits = None
        self.evaluation_metrics = None
        self.evaluation_future_metrics = None
        self.account_test_filter = None
        self.filter_threshold = None
        self.metrics_data = None
        self.metrics_future_data = None
        
        self.model_initialized = False
        print("Use create() or load() method to initialize the model.")
        
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
        print("cat_feature : {}".format(self.cat_feature))
        print("future_data_filename : {}".format(self.future_data_filename))
        print("future_target : {}".format(self.future_target))
        print("evaluation_metrics : {}".format('Not defined' if self.evaluation_metrics is None else 'Defined'))
        print("evaluation_future_metrics : {}".format('Not defined' if self.evaluation_future_metrics is None else 'Defined'))
        print("filter_threshold : {}".format(self.filter_threshold))
        print("metrics_data : {}".format('Not defined' if self.metrics_data is None else 'Defined'))
        print("metrics_future_data : {}".format('Not defined' if self.metrics_future_data is None else 'Defined'))
        print("account_filter : {}".format(self.account_filter))
        print("account_test_filter : {}".format(self.account_test_filter))
        print("-------------------------")
        
    def get_duplication_info(self):
        return self.data.groupby(['cluster','account_banner']).size()
    
    def get_account_filter(self):
        return self.data['account_banner'].unique()

    def update_info(self, analysis):
        self.test_splits = analysis.test_splitting.splits
        self.evaluation_metrics = analysis.evaluation_metrics
        self.evaluation_future_metrics = analysis.evaluation_future_metrics
        self.account_test_filter = analysis.account_test_filter
        self.filter_threshold = analysis.filter_threshold
        self.metrics_data = analysis.metrics_data
        self.metrics_future_data = analysis.metrics_future_data
        
    def create(self, params, max_evals, target_ratio_val, feature_filename, features, target, cat_feature,
               output_dir, data_filename, filter_filename, account_filter, future_data_filename, future_target,
               duplication_map):
        """
        Define the next artifacts related to input
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.params = params
        self.max_evals = max_evals or 200
        self.target_ratio_val = target_ratio_val
        self.feature_filename = feature_filename
        self.features = features
        self.target = target
        self.cat_feature = cat_feature
        self.output_dir = output_dir
        self.data_filename = data_filename
        self.filter_filename = filter_filename
        self.account_filter = account_filter
        self.future_data_filename = future_data_filename
        self.future_target = future_target
        self.duplication_map = duplication_map
        
        self.__load_dataset()
        self.__load_future_data()
        self.__load_features()
        self.__apply_account_filter()
        self.__filter_future_data()
        self.__set_clusters()
        # __set_params() runs before training the model
        self.__apply_duplication_map()
        
        self.model_initialized = True
        print("The model is created. Use AnalysisMetrics class to analyse it.")
    
    def check_status(self):
        if not self.model_initialized:
            raise Exception("Please create or load a model first")
        return True
    
    def __apply_account_filter(self):
        if self.account_filter is None:
            self.account_filter = self.data['account_banner'].unique()
            # Future data can include extra account
            self.future_data = self.future_data[self.future_data['account_banner'].isin(self.account_filter)]
        else:
            # Apply filter to historic and future data
            self.data = self.data[self.data['account_banner'].isin(self.account_filter)]
            self.future_data = self.future_data[self.future_data['account_banner'].isin(self.account_filter)]

    def __apply_duplication_map(self):
        if not self.duplication_map is None:
            if not isinstance(self.duplication_map, dict):
                raise Exception("The 'duplication_map' parameter type should be a dictionary")
            print("Duplication map:")
            new_data = self.data.copy()
            num_added = 0
            for account in self.duplication_map.keys():
                if not account in self.clusters:
                    continue
                dups = self.duplication_map[account]
                for dup in dups:
                    if dup == account:
                        continue
                    df = self.data[self.data['account_banner']==dup]
                    df['cluster'] = account
                    new_data = pd.concat([new_data, df], axis=0)
                    num_added += df.shape[0]
            # Pretty output
            for key in self.duplication_map.keys():
                print("- {}: {}".format(key, self.duplication_map[key]))
            new_data.reset_index(inplace=True, drop=True)
            print("- Initial data len: {}".format(self.data.shape[0]))
            self.data = new_data
            print("- New data len: {} (added: {})".format(self.data.shape[0], num_added))
                
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
        self.future_data = pd.read_msgpack(self.future_data_filename)
        
    def __filter_future_data(self):
        """
        Apply clearing filter to future dataset
        
        """
        print("Initial future data length: {}. Filtering...".format(self.future_data.shape[0]))
        
        # Prepare a mask that will consolidate all mask to apply filter only once in the end
        all_masks = np.zeros(len(self.future_data), dtype=bool)

        mask = self.future_data['promotion_status'].isin([30, 35])
        all_masks = np.add(all_masks, mask.values)
        print("- status of promotion: {}".format(self.future_data[mask].shape[0]))
        
        mask = ~self.future_data['original_pid'].isin(self.data['original_pid'])
        all_masks = np.add(all_masks, mask.values)
        print("- products not exist in train dataset: {}".format(self.future_data[mask].shape[0]))
        
        mask = self.future_data['exclude_yn'].isin([1, True])
        all_masks = np.add(all_masks, mask.values)
        print("- excluded flag: {}".format(self.future_data[mask].shape[0]))
        
        mask = self.future_data['baseline_units_2'] == 0.0
        all_masks = np.add(all_masks, mask.values)
        print("- zero planned baseline: {}".format(self.future_data[mask].shape[0]))
        
        mask = self.future_data['total_units_2'] == 0.0
        all_masks = np.add(all_masks, mask.values)
        print("- zero planned in-store total volume: {}".format(self.future_data[mask].shape[0]))
        
        mask = self.future_data['total_units_2'] <= self.future_data['baseline_units_2']
        all_masks = np.add(all_masks, mask.values)
        print("- planned in-store total volume lower then baseline: {}".format(self.future_data[mask].shape[0]))
        
        self.future_data = self.future_data[~all_masks]
        print("- New future data length: {}".format(self.future_data.shape[0]))
        
    def __load_features(self):
        """Load features from a file"""
        if not os.path.isfile(self.feature_filename):
            raise IOError, "Please specify a correct file with the information about features. " + \
                "File not found: " + self.feature_filename
        # Feature retrieval
        with open(self.feature_filename, 'r') as file:
            feature_info_dict = json.load(file)
        if self.features is None:
            self.features = feature_info_dict['remaining_feat'] 
        self.ordinal_features = feature_info_dict['ordinal_features'] 
        self.nominal_features = feature_info_dict['nominal_features'] 

    def __set_clusters(self):
        """
        We have the option to create a different model for each category
        (categories are based in a specific feature) or having one model
        
        """
        self.data = ClusterTransform(self.cat_feature).transform(self.data)
        self.clusters = self.data['cluster'].unique()
        self.selected_features = {}
        for cluster in self.clusters:
            self.selected_features[cluster] = self.features

    def __set_params(self):
        if self.params == None or not isinstance(self.params, dict):
            hyper = Hyperparameters(self)
            hyper.find_params()
            self.params = hyper.params

    def train(self):
        self.check_status()
        self.__set_params()
        train_x = self.data #[self.features]
        train_y = self.data[self.target]
        self.model = Pipeline([  
            ('transform', ClusterTransform(self.cat_feature)),
            ('model', MultiModel(group_col='cluster',
                                 clusters=self.clusters,
                                 params=self.params,
                                 selected_features=self.selected_features,
                                 nominals=self.nominal_features,
                                 ordinals=self.ordinal_features))
            ])
        self.model.fit(train_x, train_y)
        print("The model is trained")
        
    def train_save_to_vf(self):
        self.check_status()
        self.__set_params()
        train_x = self.data #[self.features]
        train_y = self.data[self.target]
        self.model_to_vf = MultiModel(group_col='account_banner',
                                 clusters=self.clusters,
                                 params=self.params,
                                 selected_features=self.selected_features,
                                 nominals=self.nominal_features,
                                 ordinals=self.ordinal_features)
            
        self.model_to_vf.fit(train_x, train_y)
        
        # VF related saving
        prediction_model = PredictionModel(self.model_name + '_VF', path=self.output_dir,
                                           one_hot_encode=False)
        prediction_model.model = self.model_to_vf
        # save feature names (no strictly since all the preprocessing is made being made in the pipeline)
        prediction_model.features = {key: [] for key in self.features}
        prediction_model.target = {self.target: []}

        prediction_model.ordered_column_list = sorted(prediction_model.features.keys())
        prediction_model.save()

        print("The model is trained and save (compatible with VF)")
        
    def __extended_save(self):
        # Additional information storing
        meta_path = os.path.join(os.getcwd(), self.model_name + '_ext.meta')
        meta_data = {
            'feature_filename': self.feature_filename,
            'output_dir': self.output_dir,
            'data_filename': self.data_filename,
            'filter_filename': self.filter_filename,
            'duplication_map': self.duplication_map,
            'account_test_filter': self.account_test_filter,
            'target': self.target,
            'cat_feature': self.cat_feature,
            'future_data_filename': self.future_data_filename,
            'future_target': self.future_target,
            'filter_threshold': self.filter_threshold,
            'metrics_data': self.metrics_data,
            'metrics_future_data': self.metrics_future_data,
            'account_filter': list(self.account_filter),
            'max_evals': self.max_evals,
            'target_ratio_val': self.target_ratio_val,
            'test_splits': self.test_splits,
            'evaluation_metrics': self.evaluation_metrics,
            'evaluation_future_metrics': self.evaluation_future_metrics,
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
        if not isinstance(self.model, Pipeline):
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
                self.feature_filename = meta_data.get('feature_filename')
                self.output_dir = meta_data.get('output_dir')
                self.data_filename = meta_data.get('data_filename')
                self.filter_filename = meta_data.get('filter_filename')
                self.duplication_map = meta_data.get('duplication_map')
                self.account_test_filter = meta_data.get('account_test_filter')
                self.target = meta_data.get('target')
                self.cat_feature = meta_data.get('cat_feature')
                self.future_data_filename = meta_data.get('future_data_filename')
                self.future_target = meta_data.get('future_target')
                self.filter_threshold = meta_data.get('filter_threshold')
                self.metrics_data = meta_data.get('metrics_data')
                self.metrics_future_data = meta_data.get('metrics_future_data')
                self.account_filter = meta_data.get('account_filter')
                self.max_evals = meta_data.get('max_evals')
                self.target_ratio_val = meta_data.get('target_ratio_val')
                self.test_splits = meta_data.get('test_splits')
                self.evaluation_metrics = meta_data.get('evaluation_metrics')
                self.evaluation_future_metrics = meta_data.get('evaluation_future_metrics')

    def __standard_load(self):
        prediction_model = PredictionModel(self.model_name, path=self.output_dir, one_hot_encode=False)
        self.model = prediction_model.model
        
    def load(self):
        """Load a model and artifacts from files"""
        self.__extended_load()
        self.__standard_load()
        self.__load_dataset()
        self.__load_future_data()
        self.__apply_account_filter()
        self.__filter_future_data()
        for key in self.model.named_steps['model'].selected_features:
            self.features = self.model.named_steps['model'].selected_features[key]
            break
        self.ordinal_features = self.model.named_steps['model'].ordinals 
        self.nominal_features = self.model.named_steps['model'].nominals
        self.params = self.model.named_steps['model'].params
        self.__set_clusters()
        self.__apply_duplication_map()
        
        self.model_initialized = True
        print("The model is loaded. Use AnalysisMetrics class to analyse it.")