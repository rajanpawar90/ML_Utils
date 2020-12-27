import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sklearn
from sklearn import datasets
from sklearn.datasets import load_boston
from scipy import stats

#get count of NaNs per column
def get_nan_percents(df):
    return [df.isnull().sum(), df.isnull().sum()/df.count()*100]

def drop_columns_with_na_threshold(df, p): #p is % of missing values
    df.dropna(how='all', axis=1)
    #dropped_cols=[]
    for col in df.columns:
        if df[col].isnull().sum()/df[col].count()*100 > p:
            #dropped_cols.append(col)
            df.drop([col], axis=1, inplace=True)
    return df
            
def drop_rows_with_na_threshold(df, p): #p is % of missing values
    df.dropna(how='all', axis=0, inplace=True)
    df.drop_duplicates()
    
    rows_to_be_deleted = []
    for row in range(len(df)):
        if df.iloc[row].isnull().sum()/df.iloc[row].count()*100 > p:
            rows_to_be_deleted.append(row)
    df.drop(rows_to_be_deleted, axis=0, inplace=True)
    return df

#Fill na for numericc columns with columns means
def fill_numeric_na_with_column_means(df):
    for col in df.columns:
        if df[col].dtype !=object:
            df[col].fillna(df[col].mean(), inplace=True)
    return df
            
def fill_numeric_na_with_zeros(df):
    for col in df.columns:
        if col.dtype !=object:
            df[col].fillna(0, inplace=True)
    return df

#convert target categorical variable into numerical classes
def convert_categorical_target_to_numerical(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_labelEncoded = le.fit_transform(df)
    return df_labelEncoded

# convert and drop categorical features into one hot encodings
def convert_categorical_features_to_numerical(df):
    cat_cols = []
    for col in df.columns:
        if df[col].dtype.name == 'category': 
            df_col_dummy = pd.get_dummies(df[col])
            df = pd.concat([df, df_col_dummy],axis=1)
            df.drop([col], axis=1, inplace=True)
    return df, cat_cols
    
    #return pd.get_dummies(df)
    
    #from sklearn.preprocessing import OneHotEncoder
    #oe = OneHotEncoder()
    #df_one_hot_encoded = oe.fit_transform(df)
    #return df_one_hot_encoded

# Remove outlier rows from the data which fall beyond low-high quantiles for every column
def remove_quantile_outliers_from_all_columns(df, low, high):
    cols = []
    for col in df.columns:
        if df[col].dtype != object and df[col].dtype.name != 'category':
            high_quantile = df[col].quantile(high)
            low_quantile = df[col].quantile(low)
            df = df[(df[col]<high_quantile) & (df[col]>low_quantile)]
            cols.append(col)
    return df #, cols

# Remove outliers within zscore z   
def remove_zscore_outliers_from_all_columns(df, z):
    num_cols = []
    for col in df.columns:
        if df[col].dtype != object and df[col].dtype.name != 'category':
            num_cols.append(col)
    zscores = stats.zscore(df[num_cols])
    abs_zscores = np.abs(zscores)
    return df[(abs_zscores<z).all(axis=1)]

def show_correlations(df):
    corrMatrix = df.corr()
    import seaborn as sns
    mask = np.zeros_like(corrMatrix)
    mask[np.triu_indices_from(mask)] = True
    ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(corrMatrix, annot=True, square=True)
    return corrMatrix
    
def get_correlated_features(df, thresh):
#     corrMatrix = df.corr()
#     correlated_features = []
#     for row in range(len(corrMatrix)):
#         for col in range(len(corrMatrix)):
#             if corrMatrix.iloc[row,col] >=thresh and row != col:
#                 correlated_features.append([corrMatrix.index[row],corrMatrixcol])
#     return correlated_features
    
    #efficient solution
    corrMatrix = df.corr()
    aa = corrMatrix.values
    correlated_indices = np.where((1.0>aa) &(aa>=0.2))
    c = [[corrMatrix.index[row],corrMatrix.columns[col]] for row,col in correlated_indices]
    return c
    
def remove_correlated_features(df, thresh):
    corrMatrix = df.corr()
    columns = np.full(len(corrMatrix), True, dtype=bool)
    for i in range(len(corrMatrix)):
        for j in range(i+1, len(corrMatrix)):
            if corrMatrix.iloc[i,j]>=thresh:
                if columns[j]:
                    columns[j] = False

#     for i in range(len(columns)):
#         if columns[i] == False:
#             print(i, columns[i], corrMatrix.columns[i])

# Below list comprehension works outside of function using 'is' instead of ==. find out
    discarded_cols = [corrMatrix.columns[col] for col in range(len(columns)) if columns[col] == False]
    selected_columns = corrMatrix.columns[columns] #select list of columns with value True in columns list
    df_without_correlated_features = df[selected_columns]
    return df_without_correlated_features, discarded_cols

def remove_features_based_on_p_values(df):
    import statsmodels.api as sm
    def backwardElimination(x, Y, sl, columns):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i): #size of X reduces as cols are removed and that's why '-i'
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)

        regressor_OLS.summary()
        return x, columns 
    
    SL = 0.05 
    x_df_nemericals = df.loc[:,['duration', 'campaign','pdays', 'previous', 
                                'emp.var.rate', 'cons.price.idx', 
                                'cons.conf.idx', 'euribor3m', 'nr.employed']]
    y_df_numericals = df.loc[:,['age']]
    selected_columns = x_df_nemericals.columns
    x_df_nemericals.head()
    selected_columns.shape

    #df.select_dtypes('number').columns ............to get indicex of numerical columns
    data_modeled, selected_columns = backwardElimination(x_df_nemericals.values,
                                                    y_df_numericals.values,
                                                    SL,
                                                    selected_columns)
    
    return data_modeled, selected_columns



def get_train_test_datasets(X, y):
    return True
    