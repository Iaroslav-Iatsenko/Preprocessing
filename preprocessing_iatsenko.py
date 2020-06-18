import pandas as pd
import numpy as np


def standirdize(x):
    """
    Provide z-score transforming of numpy array
    Paramerers:
    ----------
    x: Numpy array
       Array that would be transformed

    Returns
    -------
    Numpy array: z-scores numpy array
    """

    current_mean = np.mean(x)
    current_std = np.std(x)
    return (x-current_mean)/current_std


def preprocess_features(data, standardizer=standirdize):
    """ Preprocess data to standardized format with additional features

    Parameters
    ----------
    data: Pandas DataFrame
        Data with 2 columns: id_job and features. Features column contains
        comma separated features
    standardizer: function
        Function that transforms one feature based on functions logic. Current
        realization is z-score normalization
    Returns
    -------
    Pandas Dataframe
        DataFrame with next columns:
        id_job -> Integer : identifier of job
        feature_2_stand_{i} -> Double: result of standardizing i is obtained by
                               column of features
        max_feature_2_index -> Integer: index of feature with maximal value in
                               fixed row
        max_feature_2_abs_mean_diff -> Double: absolute difference of maximal
                                       feature value in a
        row with mean value of that feature

    """
    # expand features to separate columns in format feature_2_{i}
    data_expanded = data.features.str.split(',', expand=True)
    # code of features, must be the same within fixed dataframe
    features_code = data_expanded.loc[0, 0]
    # count of features could vary based on features_code
    features_count = data_expanded.shape[1]
    features_range = range(1, features_count)

    columns_names = \
        ["feature_"+str(features_code)+"_"+str(i) for i in features_range]
    columns_names.insert(0, 'features_code')
    data_expanded.columns = columns_names

    features_columns = data_expanded.columns
    data[features_columns] = data_expanded[features_columns]
    data.drop(columns=['features'], inplace=True)
    del data_expanded

    # convert to int
    for col in data.columns[1:]:
        data[col] = data[col].astype(int)

    # standardize required columns
    for i in range(1, features_count):
        col = "feature_"+str(features_code)+"_"+str(i)
        new_col = "feature_"+str(features_code)+"_stand_"+str(i)
        data[new_col] = standardizer(data[col].values)

    # get integer index of of maximum features value for fixed vacancy
    new_col_name = 'max_feature_'+str(features_code)+'_index'
    data[new_col_name] = data.iloc[:, 2:features_count+1].idxmax(axis=1)
    data[new_col_name] = data[new_col_name].apply(lambda x: x.split('_')[-1])

    # temporaty column with information of mean of feature in with fixed
    # vacancy has maximum
    def mean_column(i):
        # returns mean of column based on columns index
        col = "feature_"+str(features_code)+"_"+str(i)
        return data[col].mean()

    data['mean_of_column_with_idxmax'] = \
        data['max_feature_'+str(features_code)+'_index'].apply(mean_column)

    # absolute deviation of feature with max value in fixed vacancy from mean
    # value of that feature
    new_colname = 'max_feature_'+str(features_code)+'_abs_mean_diff'
    data[new_colname] = np.abs(data.iloc[:, 2:features_count+1].max(axis=1) -
                               data['mean_of_column_with_idxmax'])

    # preparing resulting columns
    final_columns = ['id_job']
    final_columns += ['feature_' + str(features_code) + '_stand_' + str(i)
                      for i in range(1, features_count)]
    final_columns.append('max_feature_'+str(features_code)+'_index')
    final_columns.append('max_feature_'+str(features_code)+'_abs_mean_diff')
    return data[final_columns]


def test_proc():
    # For testing purpose
    test_data = pd.read_csv('test.tsv', sep='\t')
    test_prepricessed = preprocess_features(test_data)
    test_prepricessed.to_csv('test_proc.tsv', sep='\t', index=None)
