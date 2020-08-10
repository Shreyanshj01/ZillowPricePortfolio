import pandas as pd
import numpy as np

"""
functions for the Zillow Price Project
"""



def read_in_dataset(dset, verbose=False):
    
    """Read in datasets

    Keyword arguments:
    dset : name of dataset to be read. data should be in csv format
    verbose : whether or not to print info about the dataset
    
    Returns: a pandas dataframe
    """
    
    df = pd.read_csv('unzipped_data/{0}.csv'.format(dset))
    
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the {0} dataset '.format(dset)))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.columns)
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
        
    return df


def merge_dataset(right, left):
    
    """Merge two datasets. Both need to have a common key

    Keyword arguments:
    right : Right dataframe 
    left : the dataframe 
    
    Returns:
    a pandas dataframe
    """

    train_data_merged = right.merge(left, how='left', on='parcelid')
    
    return train_data_merged



def filter_duplicate_parcels(df, random_state=0):
    """filter dataset to only include one record per parcel.

    Keyword arguments:
    df : the result of `merge_dataset`
    random_state : the random seed to be passed to the `pandas.DataFrame.sample()` method
    
    Returns:
    a pandas dataframe
    """
    counts_per_parcel = df.groupby('parcelid').size()
    more_than_one_sale = df[df.parcelid.isin(counts_per_parcel[counts_per_parcel > 1].index)]
    only_one_sale = df[df.parcelid.isin(counts_per_parcel[counts_per_parcel == 1].index)]
    reduced_df = more_than_one_sale.sample(frac=1, random_state=random_state).groupby('parcelid').head(1)
    reduced_df = pd.concat([only_one_sale, reduced_df])
    
    return reduced_df


def get_data(dset):
    
    """Create the training dataset (2016) or the test dataset (2017)

    Keyword arguments:
    dset -- a string in {train, test}
    
    Returns:
    a tuple of pandas dataframe (X) and pandas series (y)
    """
    
    year = {'train':2016, 'test':2017}[dset]
    
    train = read_in_dataset('train_{0}'.format(year))
    properties = read_in_dataset('properties_{0}'.format(year))
    merged = merge_dataset(train, properties)
    
    if dset == 'train':
        merged = filter_duplicate_parcels(merged)
    
    y = merged.pop('logerror')
    return merged, y

def mean_abs_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))