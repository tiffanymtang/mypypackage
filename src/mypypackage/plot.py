import seaborn as sns
import matplotlib.pyplot as plt


def plot_pairs(data, vars, color_by=None, corr_fontsize=10, **kwargs):
    """
    Plot pairwise relationships between variables.

    Building on seaborn's pairplot, this function shows the correlations on the
    upper triangle for continuous variables and boxplots for categorical variables.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot
    vars : list
        The variable names to plot. Should be column names in data
    color_by : str
        The variable name to color by. Should be a column name in data
    corr_fontsize : int
        The font size for the correlation values
    **kwargs
        Additional keyword arguments to pass to seaborn.pairplot

    Returns
    -------
    sns.PairGrid
        The seaborn PairGrid object

    Examples
    --------
    >>> import seaborn as sns
    >>> from mypypackage.plot import plot_pairs
    >>> data = sns.load_dataset('iris')
    >>> vars = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    >>> plot_pairs(data, vars, color_by='species')
    """

    if (color_by in vars) or (color_by is None):
        all_vars = list(vars)
    else:
        all_vars = list(vars) + [color_by]

    plt_df = data[all_vars].copy()

    # get object type for each variable under study
    cat_vars = data[all_vars].select_dtypes(include=['object', 'category']).columns
    num_vars = data[all_vars].select_dtypes(exclude=['object', 'category']).columns

    if color_by is None:
        color_mode = 'none'
    else:
        if data[color_by].dtype in ['float64', 'int64']:
            color_mode = 'continuous'
        else:
            color_mode = 'categorical'
        # convert categorical variables to numeric; otherwise, seaborn will throw an error
        for var in cat_vars:
            plt_df[var] = plt_df[var].astype("category").cat.codes

    # Plot basic pair plot
    g = sns.pairplot(
        plt_df, hue=color_by, vars=vars, corner=False, **kwargs
    )

    # compute pairwise correlations for numeric variables
    corr_matrix = plt_df[num_vars].corr()
    if color_mode == 'categorical':
        grouped_corr_matrix = plt_df.groupby(color_by)[num_vars].corr()

    # update particular subplots
    for i, row_var in enumerate(vars):
        for j, col_var in enumerate(vars):
            ax = g.axes[i, j]
            # change upper triangular plots for continuous variables to correlation values
            if (i < j) and (row_var in num_vars) and (col_var in num_vars):
                ax.set_visible(False)
                ax = g.figure.add_axes(ax.get_position(), frame_on=True)
                corr = corr_matrix.loc[row_var, col_var]
                if color_mode == 'categorical':
                    grouped_corr = grouped_corr_matrix.loc[(slice(None), row_var), col_var]
                    corr_str = "\n".join([f"{group[0]}: {val:.2f}" for group, val in grouped_corr.items()])
                    corr_str = f"Corr: {corr:.2f}\n{corr_str}"
                else:
                    corr_str = f"Corr: {corr:.2f}"
                ax.annotate(
                    corr_str,
                    xy=(0.5, 0.5), xycoords="axes fraction", ha='center', va='center', fontsize=corr_fontsize
                )
                ax.set_xticks([])
                ax.set_yticks([])
            # change diagonal plots for categorical variables to histograms
            elif (i == j) and (row_var in cat_vars):
                ax.set_visible(False)
                ax = g.figure.add_axes(ax.get_position(), frame_on=True)
                if color_mode == 'categorical':
                    sns.histplot(plt_df, x=row_var, hue=color_by, ax=ax, multiple='dodge', legend=False)
                else:
                    sns.histplot(plt_df, x=row_var, ax=ax, legend=False)
            # change diagonal plots to ignore color_by if it is a numeric variable
            elif (i == j) and (color_by in num_vars):
                ax.set_visible(False)
                ax = g.figure.add_axes(ax.get_position(), frame_on=True)
                sns.kdeplot(plt_df[row_var], ax=ax, color='black', legend=False)
            # change categorical vs categorical plots to frequency heatmaps
            elif (row_var in cat_vars) and (col_var in cat_vars):
                ax.set_visible(False)
                ax = g.figure.add_axes(ax.get_position(), frame_on=True)
                subplt_df = plt_df.groupby([row_var, col_var]).size().unstack(fill_value=0)
                sns.heatmap(subplt_df, ax=ax, cmap='Blues', annot=True, fmt='d', cbar=False)
            # change continuous vs categorical plots to boxplots
            elif row_var in cat_vars:
                ax.set_visible(False)
                ax = g.figure.add_axes(ax.get_position(), frame_on=True)
                if color_mode == 'categorical':
                    sns.boxplot(y=row_var, x=col_var, hue=color_by, data=plt_df, ax=ax, orient='h', legend=False)
                else:
                    sns.boxplot(y=row_var, x=col_var, data=plt_df, ax=ax, orient='h', legend=False)
            # change continuous vs categorical plots to boxplots
            elif col_var in cat_vars:
                ax.set_visible(False)
                ax = g.figure.add_axes(ax.get_position(), frame_on=True)
                if color_mode == 'categorical':
                    sns.boxplot(x=col_var, y=row_var, hue=color_by, data=plt_df, ax=ax, legend=False)
                else:
                    sns.boxplot(x=col_var, y=row_var, data=plt_df, ax=ax, legend=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if j != 0:
                ax.set_yticklabels([])
                ax.set_ylabel('')
            if i != (len(vars) - 1):
                ax.set_xticklabels([])
                ax.set_xlabel('')

    return g