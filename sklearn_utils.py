import matplotlib.pyplot as plt
import seaborn as sns

'''
def boxplot_regression(df_, cat_feature_, target_feature_)
- Plots sorted boxplot, how target feature varies with gradations of categorical feature

def get_correlated_attributes(df_, target_feature_, corr_threshold_):
- Selects features, that have correlated coeff "C", such that
|C| > |corr_threshold_|
- Returns a series of correlated attributes

def nan_statistics(df, nan_thresh=0.0)
- Prints out columns and nan percentage.
- Returns dictionary with columns and percentages

def visualize_datasets_distributions(
    dataframes_dict_, 
    column_numbers_, 
    grid_width_=3
)
- plots a grid of histograms of grid_width
- column_numbers_ - list of column numbers
- for each column, on the plot there are histograms
for each dataset in dataframes_dict_. To check, that train, validation 
and test data are from same distribution
- Example of usage:
visualize_datasets_distributions(
    {
        'trainval': pd.DataFrame(trainval_sample_processed),
        'test sample': pd.DataFrame(test_sample_processed),
        'test': pd.DataFrame(test_processed)
    },
    column_numbers_ = range(5),
    grid_width_=2
)

def print_model_cv_scores(sklearn_models_dict_, X_, Y_, cv_, scoring_)
- Uses sklearn cross_val_score() function
- Calculates average cross-validation score and outputs SORTED dictionary of results
- Returns sorted dictionaty with models names and their average CV scores
- Example of usage:
_ = print_model_cv_scores(
    sklearn_models_dict_={
        model_name: model.model for model_name, model in all_models.items()
    },
    X_=X_train_val,
    Y_=Y_train_val,
    cv_=7,
    scoring_='neg_mean_squared_error'
)


'''

def boxplot_regression(df_, cat_feature_, target_feature_):
    subset = df_[[cat_feature_, target_feature_]]
    s = subset.groupby([cat_feature_]).median().sort_values(by=target_feature_)
    
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(x=cat_feature_, data=df_, stat='percent')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=cat_feature_, y=target_feature_, data=df_, order=s.index)


def get_correlations(df_, target_feature_, ascending_=False):
    cm = df_.corr()
    return cm[target_feature_].sort_values(ascending=ascending_)


def get_correlated_attributes(df_, target_feature_, corr_threshold_):
    '''
    Selects features, that have correlated coeff "C", such that
    |C| > |corr_threshold_|
    '''
    corrs = get_correlations(df_, target_feature_)
    return corrs.loc[
        (corrs >= abs(corr_threshold_)) |
        (corrs <= -abs(corr_threshold_))
    ]


def nan_percentage(df, colname):
    return (df[colname].isnull().sum() / df.shape[0]) * 100


def nan_statistics(df, nan_thresh=0.0):
    res = {}
    nan_cols = df.loc[:, df.isna().any()].columns
    for col in nan_cols:
        res[col] = nan_percentage(df, col)
    print(f'Col -- Nan percentage')
    for key, val in sorted(res.items(), key=lambda item: item[1], reverse=True):
        if val >= nan_thresh * 100:
            print(key, val)
        else:
            del res[key]
    return res


def visualize_datasets_distributions(
    dataframes_dict_,
    column_numbers_,
    grid_width_=3
):
    
    n_plots = len(column_numbers_)
    if n_plots % grid_width_ == 0:
        grid_height = int(n_plots / grid_width_)
    else:
        grid_height = int(n_plots / grid_width_) + 1
        
    _, ax = plt.subplots(grid_height, grid_width_, figsize=(10, 10))

    for i in range(grid_height):
        for j in range(grid_width_):
            cur_column_number = i * (grid_width_) + j
            
            if cur_column_number > n_plots:
                return

            columns_data = {}
            for dataset_name, df in dataframes_dict_.items():
                columns_data[dataset_name] = df.values[:, cur_column_number]
            
            
            for dataset_name, data in columns_data.items():
                ax[i, j].hist(data, density=True, alpha=0.3, label=dataset_name)

            ax[i, j].set_title(f'Column {cur_column_number}')
            ax[i, j].legend()


def print_model_cv_scores(sklearn_models_dict_, X_, Y_, cv_, scoring_):
    res = {}
    for name, model in sklearn_models_dict_.items():
        scores = cross_val_score(
            model,
            X_,
            Y_,
            cv=cv_,
            scoring=scoring_
        )
        res[name] = scores
    
    # Sort the dict
    sorted_res = {
        k:v for \
        k, v in sorted(res.items(), key = lambda item: np.mean(item[1]))
    }
    for model_name, scores in sorted_res.items():
        print(f'Model: {model_name}, mean: {np.mean(scores)}, std: {np.std(scores)}')

    return sorted_res
