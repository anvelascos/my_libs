import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

import hydrobasics as hb


def fn_setfeat(dict_qlsel, list_feat, dict_total_feat, t=1):
    """
    This function combines the set of features from a list with series names and lags.
    :param dict_qlsel:
    :param list_feat: dictionaries list with series that have series names and the lags respective.
    :param dict_total_feat:
    :param t:
    :return:
    """
    dict_feat = {}
    xls_output = pd.ExcelWriter('feat_sel.xlsx')
    for sta_name in dict_qlsel:
        df_total_sta = dict_total_feat[sta_name]
        df_sel_feat = pd.DataFrame()
        for feat_group in list_feat:
            feat_sta = feat_group[sta_name]
            if not isinstance(feat_sta, dict):
                for feat_single in feat_sta.index:
                    list_single = [x for x in feat_sta.loc[feat_single] if x >= t]
                    df_series_lag = hb.fn_lagsr(df_total_sta[feat_single], list_single)
                    df_sel_feat[df_series_lag.columns] = df_series_lag
            else:
                for feat_group_par in feat_sta:
                    feat_sta_par = feat_sta[feat_group_par]
                    for feat_single_par in feat_sta_par.index:
                        list_single = [x for x in feat_sta_par.loc[feat_single_par] if x >= t]
                        df_series_lag = hb.fn_lagsr(df_total_sta[feat_single_par], list_single)
                        df_sel_feat[df_series_lag.columns] = df_series_lag
        dict_feat[sta_name] = df_sel_feat
        df_sel_feat.to_excel(xls_output, sheet_name=sta_name, merge_cells=False)
    xls_output.save()
    return dict_feat


def fn_regrem(x, y, alpha=.05, retsum=False, max_cn=300.):
    """
    This function performs a linear regression (OLS) by removing non significant features from the initial list.
    If there is not adjust, the function returns a Series with an unique index ('const') with value zero.
    :param x: Features matrix.
    :param y: Response variable.
    :param alpha: Significance level.
    :param retsum: Returns OLS summary.
    :param max_cn: Maximum Condition Number allowed.
    :return: OLS params, OLS summary.
    """
    cond = True
    pvalues = []

    while cond:

        try:
            mod = sm.OLS(y, x, hasconstant=('const' in x.columns)).fit()
            pvalues = pd.Series(mod.pvalues, index=x.columns)
            cond = (len(pvalues) > 0) & ((pvalues > alpha).any() or (mod.condition_number > max_cn))

            if cond:
                ix_rem = pvalues.idxmax()
                # pvalues.drop(ix_rem, inplace=True)
                x.drop(ix_rem, axis=1, inplace=True)

        except Exception, e:
            print('No adjust, zero will be returned. {}'.format(e))
            pvalues = []
            cond = False

    if len(pvalues) == 0:

        if retsum:
            return None, '\nNo adjust.\n'

        else:
            return None

    else:

        if retsum:
            return mod.params, mod.summary()

        else:
            return mod.params


def fn_reglin(x, y, alpha=0.05, hasconst=True):
    """
    this function implement a linear regression and returns
    :type x: object
    :param x: matrix with predictors (OAI, PT, QL, ...)
    :param y: endogen variable (PT, QL...)
    :param alpha: statistical significance
    :param hasconst: does regression have constant?
    :return: a list with [r2, condition_number, list with variables with pvalue < alpha]
    """
    mod = sm.OLS(endog=y, exog=x, hasconst=hasconst, alpha=alpha).fit()
    sr_pvalues = mod.pvalues
    if hasconst:
        lst_ok = sr_pvalues[sr_pvalues < alpha].index[:-1].values  # Remove Const from the list
    else:
        lst_ok = sr_pvalues[sr_pvalues < alpha].values  # Don't remove Const from the list, because it is not there.
    return [mod.rsquared, mod.condition_number, lst_ok]


def fn_regadd(x, y, alpha=.05, retsum=False, max_cn=10., gain_r2=.01):
    """
    This function performs a linear regression (OLS) by adding significant features to the initial list.
    If there is not adjust, the function returns a Series with an unique index ('const') with value zero.
    :param x_try: Features matrix.
    :param y: Response variable.
    :param alpha: Significance level.
    :param retsum: Returns OLS summary.
    :param max_cn: Maximum Condition Number allowed.
    :return: OLS params, OLS summary.
    """
    col_feat = x.columns
    df_selected = pd.DataFrame(index=x.index)
    r2_sel = 0

    while col_feat.size > 0:
        df_results = pd.DataFrame(index=col_feat, columns=['r2', 'cn', 'pvalue_ok'])

        for feat in col_feat:  # iterate over each climate index
            # Prepare data for regression
            x_try = df_selected.copy()
            x_try[feat] = x[feat]
            # x_try = sm.add_constant(x_try, prepend=False)
            # run function and save results in results dataframe

            mod = sm.OLS(y, x_try, hasconst=False).fit()
            pvalues = mod.pvalues
            df_results.loc[feat, 'r2'] = mod.rsquared
            df_results.loc[feat, 'cn'] = mod.condition_number
            df_results.loc[feat, 'pvalue_ok'] = pvalues[pvalues < alpha].index

        df_results.sort_values(['r2', 'cn'], ascending=[False, True], inplace=True)
        sel_oai = df_results.iloc[0].name
        pvalues_ok = df_results.loc[sel_oai, 'pvalue_ok']
        r2_try = df_results.loc[sel_oai, 'r2']
        cn_try = df_results.loc[sel_oai, 'cn']

        if sel_oai in pvalues_ok and len(pvalues_ok) > len(df_selected.columns) and (r2_try - r2_sel) > gain_r2 and cn_try < max_cn:
            df_selected[sel_oai] = x[sel_oai]
            r2_sel = r2_try

        col_feat = col_feat[col_feat != sel_oai]

    x_sel = sm.add_constant(df_selected, prepend=False)
    mod_sel = sm.OLS(endog=y, exog=x_sel, hasconst=True).fit()

    pars = mod_sel.params
    if mod_sel.pvalues['const'] > alpha:
        pars = pars.drop('const')

    if retsum:
        return pars, mod_sel.summary()
    else:
        return pars


def clao_regrem(date_forecast, df_feat, sr_target, w, applypca=False):
    ix_train = pd.Index(pd.date_range(date_forecast - pd.DateOffset(months=w), periods=w + 1, freq='MS'), name='Date')
    df_feat_train = hb.fn_std(df_feat.loc[ix_train])
    y_train = sr_target.loc[ix_train][:-1]

    if applypca:
        pca = PCA(n_components=.85)
        df_comp = pd.DataFrame(pca.fit_transform(df_feat_train),
                               index=ix_train, columns=range(1, pca.n_components_ + 1))
        x_train = sm.add_constant(df_comp, prepend=False)[:-1]
        x_fore = df_comp.loc[date_forecast]
    else:
        x_train = sm.add_constant(df_feat_train, prepend=False)[:-1]
        x_fore = df_feat.loc[date_forecast]
        size = len(y_train) - 2
        cols = x_train.columns.drop('const')
        corr = pd.Series([abs(y_train.corr(x_train[sr])) for sr in cols], index=cols)
        ix_new = corr.sort_values(inplace=False, ascending=False).index[:size]
        x_train = x_train[ix_new]

    coefs = fn_regrem(x_train, y_train)
    x_fore = x_fore[coefs.index]
    if 'const' in coefs.index:
        x_fore.loc['const'] = 1
    y_fore = (x_fore * coefs).sum()

    return y_fore


def core_informativity(y, x):
    # model = OLS(y, zip(*x))
    model = sm.OLS(y, x)
    results = model.fit()
    r2 = results.rsquared
    informativity = .5 * np.log(1 / (1 - r2))
    return informativity


def fn_informativity(x, y, alpha=.05, reqinfo=.85):
    sr_corr = x.corrwith(y).abs().sort_values(ascending=False)
    sr_n = pd.notnull(x).sum()
    rteo = hb.fn_rteo(sr_n, alpha)
    sr_corr = sr_corr[sr_corr > rteo]
    sel_stations = sr_corr.index

    if len(sel_stations) == 0:
        print("Can't find features by informativity method.")
        return None

    sr_info = pd.Series(index=sr_corr.index, name=y.name)

    for i_station, sel_station in enumerate(sel_stations, 1):
        try:
            sr_info.loc[sel_station] = core_informativity(y=y, x=x[sel_stations[:i_station]])

        except Exception as e:
            print("Error: informativity function. {}".format(e))
            return None

    sr_diff = sr_info.diff()
    sr_diff.loc[sel_stations[0]] = sr_info.loc[sel_stations[0]]
    sr_diff.sort_values(ascending=False, inplace=True)
    sr_diff /= sr_info.max()
    sr_cumdiff = sr_diff.cumsum()
    ix_sel = sr_cumdiff.index[:len(sr_cumdiff[sr_cumdiff < reqinfo]) + 1]
    return ix_sel
