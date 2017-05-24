import pandas as pd
import scipy.stats as ss


def doublemadsfrommedian(df_input, axis=0):
    """
    This function calculates the MADs from median (Double MAD).
    More information: http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers
    :param df_input:
    :param axis:
    :return:
    :type axis: int
    :type df_input: pd.DataFrame
    """
    median = df_input.median(axis=axis)

    def doublemad():
        abs_dev = (df_input - median).abs()
        left_mad = (abs_dev[df_input < median]).median(axis=axis)
        right_mad = (abs_dev[df_input > median]).median(axis=axis)
        return left_mad, right_mad

    two_sided_mad = doublemad()
    ix_input = df_input.index
    x_mad = pd.DataFrame(index=ix_input, columns=df_input.columns)
    x_mad[df_input < median] = pd.DataFrame(data=[two_sided_mad[0].values] * len(ix_input), index=ix_input,
                                            columns=df_input.columns)
    x_mad[df_input > median] = pd.DataFrame(data=[two_sided_mad[1].values] * len(ix_input), index=ix_input,
                                            columns=df_input.columns)
    mad_distance = (df_input - median).abs() / x_mad
    mad_distance[df_input == median] = 0
    return mad_distance


def singlemadsfrommedian(df_input, axis=0, c=1):
    """
    This function calculates the MADs from median (Single MAD).
    More information: http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers
    :param df_input:
    :param axis:
    :param c:
    :return:
    :type axis: int
    :type df_input: pd.DataFrame
    :type c: int
    """
    median = df_input.median(axis=axis)

    def singlemad():
        mad = ((df_input - median).abs() / c).median(axis=axis)
        return mad

    x_mad = singlemad()
    mad_distance = (df_input - median).abs() / x_mad
    return mad_distance


def fn_grubbs(df_input, alpha=0.05, two_tail=True):
    """
    This function applies the Grubbs' Test for outliers in a dataframe and returns two dataframes, the first one
    without outliers and the second one just for the outliers
    :param df_input: Pandas dataframe with series to test.
    :param alpha: Significance level [1% as default].
    :param two_tail: Two tailed distribution [True as default].
    :return: tuple with two dataframes, the first one without outliers and the second one just for outliers.
    """

    if isinstance(df_input, pd.Series):
        df_input = pd.DataFrame(df_input)

    df_try = df_input.copy()
    df_output = pd.DataFrame(index=df_input.index, columns=df_input.columns)
    df_outliers = pd.DataFrame(data=0, index=df_input.index, columns=df_input.columns)

    if two_tail:
        alpha /= 2

    i = 0
    while not df_outliers.isnull().values.all():
        mean = df_try.mean()
        std = df_try.std()
        n = len(df_try)
        tcrit = ss.t.ppf(1 - (alpha / (2 * n)), n - 2)
        zcrit = (n - 1) * tcrit / (n * (n - 2 + tcrit ** 2)) ** .5
        df_outliers = df_try.where(((df_try - mean) / std) > zcrit)
        df_output.update(df_input[df_outliers.isnull() == False])
        df_try = df_try[df_outliers.isnull()]
        i += 1
        print i

    return df_try, df_output
