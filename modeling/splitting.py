import numpy as np
import pandas as pd

class Splitting(object):
    
    def __init__(self, splits=None, data=None, number_tests=100, target_ratio=0.2):
        self.splits = splits
        self.number_tests = number_tests
        self.target_ratio = target_ratio
        if data is None and self.splits is None:
            raise Exception("If splits is not defined that data should be provided!")
        elif self.splits is None:
            # Get required fields
            # if data contains duplicates, remove them
            self.data = data[['account_banner', 'yearweek', 'promotion_technical_id', 'original_pid']].drop_duplicates()
            self.__prepare_splits()
        elif not isinstance(self.splits, dict):
            raise Exception("Splits should be a dictionary")
        elif len(self.splits) != number_tests:
            raise Exception("Length of splits ({}) is not equal to number_tests ({})!".format(len(self.splits), number_tests))
        
    def get_split(self, seed, data):
        """
        Get split according to saved information about splits as every 'seed' step
        
        """
        test = data[data['promotion_technical_id'].isin(self.splits[str(seed)])]
        train = data[~data['promotion_technical_id'].isin(self.splits[str(seed)])]

        return train, test
        
    def __prepare_splits(self):
        """
        Prepare split for each from num_target splits
        
        """
        print("Preparing splits...")
        self.splits = dict()
        for seed in range(self.number_tests):
            df = self.data
            test_indexes = self.__indexes_for_split(df, seed)
            mask = df.index.isin(test_indexes)
            test = df[mask]
            train = df[~mask]
            # Exclude products which exists only in test
            test = test[test['original_pid'].isin(train['original_pid'])]
            # Adding to splist
            self.splits[str(seed)] = list(test['promotion_technical_id'].unique())
            
    def __indexes_for_split(self, df, seed):
        # get indexes for part of dataset
        split_indexes = []
        account_list = list(df['account_banner'].unique())

        for account in account_list:
            split_indexes += self.__indexes_for_split_by_account(df, account, seed)

        return split_indexes

    def __indexes_for_split_by_account(self, df, account, seed):
        # get indexes for part of dataset
        total_promo_qty = df[df['account_banner']==account].promotion_technical_id.nunique()
        split_promo_qty = 0
        split_indexes = []
        ratio = 0
        while ratio < self.target_ratio:
            indexes, promo_qty = self.__random_promo_by_account(df, split_indexes, account, seed)
            split_indexes += indexes
            split_promo_qty += promo_qty
            ratio = float(split_promo_qty) / float(total_promo_qty)    
        return split_indexes


    def __random_promo_by_account(self, df, indexes, account, seed):
        """
        Get the random indexes of promo activities by account excluding indexes in indexes
        
        """
        latest_indexes = list()
        promo_qty = 0
        # Remove known indexes
        mask = df.index.isin(indexes)
        df = df[~mask & (df['account_banner']==account)]
        yearweeks = df.yearweek.unique()
        np.random.seed(seed)
        rand_yearweek = np.random.choice(yearweeks, 1)[0]
        latest_indexes += df[df['yearweek']==rand_yearweek].index.to_list()
        promo_qty += df[df['yearweek']==rand_yearweek].promotion_technical_id.nunique()
        indexes += latest_indexes
        return indexes, promo_qty


class LastSplitting(object):
    """
    
    """
    
    def __init__(self, data, target_ratio=0.2):
        # if data contains duplicates, remove them
        self.data = data[['account_banner', 'yearweek', 'promotion_technical_id', 'original_pid']].drop_duplicates()
        self.target_ratio = target_ratio

    def get_split(self):
        # get indexes for part of dataset
        total_promo_qty = self.data.promotion_technical_id.nunique()
        split_promo_qty = 0
        split_indexes = []
        ratio = 0
        while ratio < self.target_ratio:
            indexes, promo_qty = self.__last_promo(self.data, split_indexes)
            split_indexes += indexes
            split_promo_qty += promo_qty
            ratio = float(split_promo_qty) / float(total_promo_qty)
        return split_indexes

    def __last_promo(self, df, indexes):
        """
        Get the latest indexes of promo activities excluding indexes in indexes

        """
        latest_indexes = list()
        promo_qty = 0
        # Remove known indexes & get the last week
        mask = df.index.isin(indexes)
        df = df[~mask]
        max_week = df['yearweek'].max()
        latest_indexes += df[df['yearweek']==max_week].index.to_list()
        promo_qty = df[df['yearweek']==max_week].promotion_technical_id.nunique()
        indexes += latest_indexes
        return indexes, promo_qty