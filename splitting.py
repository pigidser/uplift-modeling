import numpy as np
import pandas as pd

class Splitting(object):
    
    def __init__(self):
        self.split = None

    def random_promo_by_account(self, df, indexes, account, seed):
        """
        Get the latest indexes of promo activities by account excluding indexes in indexes
        
        """
        latest_indexes = list()
        promo_qty = 0
        # Remove known indexes
        mask = df.index.isin(indexes)
        df = df[~mask & (df['account_banner']==account)]
    #     max_week = df['week'].max()
        weeks = df.week.unique()
        np.random.seed(seed) # 2 is okay
        rand_week = np.random.choice(weeks, 1)[0]
        latest_indexes += df[df['week']==rand_week].index.to_list()
        promo_qty += df[df['week']==rand_week].promotion_technical_id.nunique()
        indexes += latest_indexes
        return indexes, promo_qty

    def indexes_for_split_by_account(self, df, target_ratio, account, seed):
        # get indexes for part of dataset
        total_promo_qty = df[df['account_banner']==account].promotion_technical_id.nunique()
        split_promo_qty = 0
        split_indexes = []
        ratio = 0
        while ratio < target_ratio:
            indexes, promo_qty = self.random_promo_by_account(df, split_indexes, account, seed)
            split_indexes += indexes
            split_promo_qty += promo_qty
            ratio = float(split_promo_qty) / float(total_promo_qty)    
        return split_indexes


    def indexes_for_split(self, df, target_ratio, seed):
        # get indexes for part of dataset
        split_indexes = []
        account_list = list(df['account_banner'].unique())

        for account in account_list:
            split_indexes += self.indexes_for_split_by_account(df, target_ratio, account, seed)

        return split_indexes
    
    def split(self, seed, init_train, target_ratio_test):
        """
        Custom split dataset
        
        """
        train = init_train.copy()
        test_indexes = self.indexes_for_split(train, target_ratio_test, seed)
        mask = train.index.isin(test_indexes)
        test = train[mask]
        train = train[~mask]

        # Exclude products which exists only in test
        test = test[test['original_pid'].isin(train.original_pid)]

        return train, test