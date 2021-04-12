import os
import numpy as np
import pandas as pd
import json

from vf_portalytics.multi_model import MultiModel
from vf_portalytics.model import PredictionModel

from splitting import LastSplitting
from hyperparams import Hyperparameters
from analysis import AnalysisMetrics
from transforms import ClusterTransform

from sklearn.pipeline import Pipeline

class Model(object):
    
    """
    This class manipulates with model: creates, trains, saves, loads.
        
    # Declare a model
    model = Model(model_name='test')
    
    # Creating the model from a scratch
    model.create(
        params=None,
        max_evals=10,
        target_ratio_param=None,
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

    
    # If existing params is used, max_evals not used.
    params=params,
    max_evals=None

    # Features are loaded from feature_filename, or can be defined by features parameter (list of features)

    # Finally the model should be trained:
    model.train()
    
    # After training, the model can be saved:
    model.save()
    model.save(model_name='new_model_name')

    # Loading model:
    model = Model(model_name='test')
    model.load()
    
    """

    def __init__(self, model_name, show_tips=True):
        """
        
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.model_name = model_name
        self.show_tips = show_tips
        self.target_ratio_test = None
        self.test_splits = None
        self.account_test_filter = None
        self.confidence_threshold = None
        self.evaluation_metrics = None
        self.evaluation_deferred_metrics = None
        self.evaluation_future_metrics = None
        self.confidence_metrics = None
        self.confidence_deferred_metrics = None
        self.confidence_future_metrics = None
        
        self.model_initialized = False
        self.model_trained = False
        self.__tips_create_or_load()
        
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
        print("confidence_threshold : {}".format(self.confidence_threshold))
        print("confidence_metrics : {}".format('Not defined' if self.confidence_metrics is None else 'Defined'))
        print("confidence_future_metrics : {}".format('Not defined' if self.confidence_future_metrics is None else 'Defined'))
        print("account_filter : {}".format(self.account_filter))
        print("account_test_filter : {}".format(self.account_test_filter))
        print("-------------------------")
        
    def get_duplication_info(self):
        return self.data.groupby(['cluster','account_banner']).size()
    
    def get_account_filter(self):
        return self.data['account_banner'].unique()

    def create(self, params, max_evals, target_ratio_param, target_ratio_defer, feature_filename,
               features, target, cat_feature, output_dir, data_filename, filter_filename, account_filter,
               future_data_filename, future_target, duplication_map):
        """
        Define the next artifacts related to input
        """
#         self.logger = logging.getLogger('sales_support_helper_application.' + __name__)
        self.params = params
        self.max_evals = max_evals or 200
        self.target_ratio_param = target_ratio_param or 0.30
        self.target_ratio_defer = target_ratio_defer or 0.20
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
        self.__deferred_split()
        self.__filter_future_data()
        self.__set_clusters()
        # __set_params() runs before training the model
        self.__apply_duplication_map()
        
        self.model_initialized = True
        self.__tips_training()
    
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
        # VF AI environment does not support planned baseline. The model always
        # should be trained with baseline_units. As we need planned baseline,
        # we make sustitution.
        df['baseline_units'] = df['baseline_units_2']
        self.data = df
        
    def __deferred_split(self):
        # Create deferred sample
        split = LastSplitting(self.data, self.target_ratio_defer)
        indexes = split.get_split()
        mask = self.data.index.isin(indexes)
        self.deferred_data = self.data[mask]
        self.data = self.data[~mask]
        
    def __load_future_data(self):
        if not os.path.isfile(self.future_data_filename):
            raise IOError, "Please specify a correct file with train dataset. File not found: " \
                + self.future_data_filename
        df = pd.read_msgpack(self.future_data_filename)
        # VF AI environment does not support planned baseline. The model always
        # should be trained with baseline_units. As we need planned baseline,
        # we make sustitution.
        df['baseline_units'] = df['baseline_units_2']
        self.future_data = df
        
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
        self.model_trained = True
        print("The model is trained")
        self.__tips_save()
        self.__tips_analysis()

    def train_save_to_vf(self):
        self.check_status()
        self.__set_params()
        train_x = self.data
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
        if self.account_test_filter is None:
            self.account_test_filter = self.data['account_banner'].unique()
        meta_path = os.path.join(os.getcwd(), self.model_name + '_ext.meta')
        meta_data = {
            'feature_filename': self.feature_filename,
            'output_dir': self.output_dir,
            'data_filename': self.data_filename,
            'filter_filename': self.filter_filename,
            'duplication_map': self.duplication_map,
            'target': self.target,
            'cat_feature': self.cat_feature,
            'future_data_filename': self.future_data_filename,
            'future_target': self.future_target,
            'confidence_threshold': self.confidence_threshold,
            'evaluation_metrics': self.evaluation_metrics,
            'evaluation_deferred_metrics': self.evaluation_deferred_metrics,
            'evaluation_future_metrics': self.evaluation_future_metrics,
            'confidence_metrics': self.confidence_metrics,
            'confidence_deferred_metrics': self.confidence_deferred_metrics,
            'confidence_future_metrics': self.confidence_future_metrics,
            'account_filter': list(self.account_filter),
            'account_test_filter': list(self.account_test_filter),
            'max_evals': self.max_evals,
            'target_ratio_param': self.target_ratio_param,
            'target_ratio_defer': self.target_ratio_defer,
            'target_ratio_test': self.target_ratio_test,
            'test_splits': self.test_splits,
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

    def __meta_path(self):
        return os.path.join(os.getcwd(), self.model_name + '_ext.meta')

    def __model_saved(self):
        meta_path = self.__meta_path()
        return os.path.isfile(meta_path)
        
    def __extended_load(self):
        meta_path = self.__meta_path()
        if not self.__model_saved():
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
                self.target = meta_data.get('target')
                self.cat_feature = meta_data.get('cat_feature')
                self.future_data_filename = meta_data.get('future_data_filename')
                self.future_target = meta_data.get('future_target')
                self.confidence_threshold = meta_data.get('confidence_threshold')
                self.evaluation_metrics = meta_data.get('evaluation_metrics')
                self.evaluation_deferred_metrics = meta_data.get('evaluation_deferred_metrics')
                self.evaluation_future_metrics = meta_data.get('evaluation_future_metrics')
                self.confidence_metrics = meta_data.get('confidence_metrics')
                self.confidence_deferred_metrics = meta_data.get('confidence_deferred_metrics')
                self.confidence_future_metrics = meta_data.get('confidence_future_metrics')
                self.account_filter = meta_data.get('account_filter')
                self.account_test_filter = meta_data.get('account_test_filter')
                self.max_evals = meta_data.get('max_evals')
                self.target_ratio_param = meta_data.get('target_ratio_param')
                self.target_ratio_defer = meta_data.get('target_ratio_defer')
                self.target_ratio_test = meta_data.get('target_ratio_test')
                self.test_splits = meta_data.get('test_splits')
                
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
        self.__deferred_split()
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
        self.model_trained = True
        print("The model is loaded.")
        self.__tips_analysis()
        
    def __tips_create_or_load(self):
        if not self.show_tips:
            return
        if self.__model_saved():
            print("Tip: the model exists. You can load it:")
            print("model.load()")
            print("or use create() method")
        else:
            print("Tip: the model does not exist. Use create() method:")
            print("model.create(")
            print("    params=None,")
            print("    max_evals=10,")
            print("    target_ratio_param=None,")
            print("    feature_filename='./outputs/im_feature_info_dict_mars_ru_20210212.txt',")
            print("    features=None,")
            print("    target='total_units',")
            print("    cat_feature='account_banner',")
            print("    output_dir='outputs',")
            print("    data_filename='../data/20210212_mars_ru_prod_trainset.msgpack',")
            print("    filter_filename='./outputs/im_data_retrieval-v6-20210212.txt',")
            print("    account_filter=['5Pyaterochka (X5)','Lenta','Dixy','Okey Group','Magnit'],")
            print("    future_data_filename='../data/20210212_mars_ru_prod_futureset.msgpack',")
            print("    future_target='total_units_2',")
            print("    duplication_map=None)")
            
    def __tips_training(self):
        if not self.show_tips:
            return
        print("Tip: for training the model, use:")
        print("model.train()")
        
    def __tips_save(self):
        if not self.show_tips:
            return
        print("Tip: to save the model, use:")
        print("model.save()")
        
    def __tips_analysis(self):
        if not self.show_tips:
            return
        if self.evaluation_metrics:
            print("Tip: the model was evaluated. Use AnalysisMetrics() class to analyse it:")
            print("analysis_metrics = AnalysisMetrics(model=model)")
        else:
            print("Tip: the model is not evaluated. Use AnalysisMetrics() class to analyse it:")
            print("analysis_metrics = AnalysisMetrics(")
            print("    model=model,")
            print("    number_tests=100,")
            print("    confidence_threshold=300,")
            print("    account_test_filter=['5Pyaterochka (X5)','Lenta','Dixy','Okey Group'])")