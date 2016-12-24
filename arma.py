import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics import tsaplots
from statsmodels.tsa import stattools
from statsmodels.tsa.arima_model import ARMA

import my_libs.metrics as metrics
import my_libs.eof as eof
import my_libs.hydrobasics as hb


# plt.style.use('ggplot')


def test_stationarity(timeseries, printval=False, retval=False):
    """
    This function performs the Dickey-Fuller Test.
    :param timeseries: data time series.
    :param printval:
    :param retval:
    :return:
    """
    dftest = stattools.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    dfresults = pd.DataFrame(columns=['Value', 'Stationary'])

    for key, value in dftest[4].items():
        dfresults.loc['Critical Value (%s)' % key] = [value, dfoutput[0] < value]

    if printval:
        print('Results of Dickey-Fuller Test:')
        print(dfoutput.to_string())
        print(dfresults)

    if retval:
        return dfoutput, dfresults


def core_arma(y, df_feat=None, par='QL', savefigs=False, printval=False, max_ar=4, max_ma=2):
    """

    :param y:
    :param df_feat:
    :param par:
    :param savefigs:
    :param printval:
    :return:
    """
    try:
        test_stationarity(y, printval=printval)

    except Exception as e:
        print("Dickey-Fuller Test can't be performed. {}".format(e))

    name_st = y.name

    if savefigs:
        # Plot ACF and PACF
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        tsaplots.plot_acf(y, lags=60, ax=ax1)
        ax1.set_title('Autocorrelation Function')
        tsaplots.plot_pacf(y, lags=60, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function')
        plt.suptitle('Autocorrelation Graphs for {station} Series'.format(station=name_st), fontsize=14)
        name_fig = 'figs/{par}_{sta}_acf_pacf'.format(par=par, sta=name_st)
        plt.savefig(name_fig)
        plt.close()

    # res = stattools.arma_order_select_ic(y)
    # best_pq = list({res.bic_min_order})
    res = stattools.arma_order_select_ic(y, max_ar=max_ar, max_ma=max_ma, ic=['aic', 'bic', 'hqic'])
    best_pq = list({res.aic_min_order, res.bic_min_order, res.hqic_min_order})

    if printval:
        print('BIC-order:  {}'.format(res.bic_min_order))
        print('AIC-order:  {}'.format(res.aic_min_order))
        print('HQIC-order: {}'.format(res.hqic_min_order))

    # Run simulation
    dict_res = {}
    df_met = pd.DataFrame(index=['r2', 'ssd', 'rmse'])

    for pq_order in best_pq:
        if printval:
            print('\nARMA Order: {}'.format(pq_order))

        try:
            if df_feat is not None:
                arma = ARMA(endog=y, order=list(pq_order), exog=df_feat)

            else:
                arma = ARMA(endog=y, order=list(pq_order))

            res_arma = arma.fit(solver='cg', full_output=True, disp=False)  # disp=-1)
            y_sim = res_arma.fittedvalues
            # plt.plot_adjust(y, color='blue')
            # plt.plot_adjust(y_sim, color='red')
            # plt.show(block=False)
            # plt.close()
            mt_arma = metrics.Metrics(y, y_sim)
            sr_met = pd.Series(index=['r2', 'ssd', 'rmse'], name='{st}_{pq}'.format(st=name_st, pq=pq_order))
            sr_met.loc['r2'] = mt_arma.r2
            sr_met.loc['ssd'] = mt_arma.ssd
            sr_met.loc['rmse'] = mt_arma.rmse

            if printval:
                print(sr_met.to_string())

            dict_res[pq_order] = res_arma
            df_met[pq_order] = sr_met

        except Exception as e:
            if printval:
                print("The ARMA{} model couldn't be performed. {}".format(pq_order, e))

            pass

    df_met.loc['r2'] = df_met.loc['r2'] == df_met.loc['r2'].max()
    df_met.loc['ssd'] = df_met.loc['ssd'] == df_met.loc['ssd'].min()
    df_met.loc['rmse'] = df_met.loc['rmse'] == df_met.loc['rmse'].min()
    ix_sel = df_met.sum()[df_met.sum() == df_met.sum().max()].index[0]  # Be careful with this line

    print('Selected Model: ARMA{}'.format(ix_sel))

    return dict_res[ix_sel]


def core_arma_v0(y, df_feat=None, par='QL', savefigs=False, printval=False):
    """

    :param y:
    :param df_feat:
    :param par:
    :param savefigs:
    :param printval:
    :return:
    """
    try:
        test_stationarity(y, printval=printval)
    except Exception as e:
        print("Dickey-Fuller Test can't be performed. {}".format(e))

    name_st = y.name

    if savefigs:
        # Plot ACF and PACF
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        tsaplots.plot_acf(y, lags=24, ax=ax1)
        ax1.set_title('Autocorrelation Function')
        tsaplots.plot_pacf(y, lags=24, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function')
        plt.suptitle('Autocorrelation Graphs for {station} Series'.format(station=name_st), fontsize=14)
        name_fig = 'figs/{par}_{sta}_acf_pacf'.format(par=par, sta=name_st)
        plt.savefig(name_fig)
        plt.close()

    res = stattools.arma_order_select_ic(y, ic=['aic', 'bic', 'hqic'])
    best_pq = list({res.aic_min_order, res.bic_min_order, res.hqic_min_order})

    if printval:
        print('AIC-order:  {}'.format(res.aic_min_order))
        print('BIC-order:  {}'.format(res.bic_min_order))
        print('HQIC-order: {}'.format(res.hqic_min_order))

    # Run simulation
    dict_res = {}
    df_met = pd.DataFrame(index=['r2', 'ssd', 'rmse'])
    for pq_order in best_pq:
        if printval:
            print('\nARMA Order: {}'.format(pq_order))
        try:
            if df_feat is not None:
                arma = ARMA(endog=y, order=list(pq_order), exog=df_feat)
            else:
                arma = ARMA(endog=y, order=list(pq_order))

            res_arma = arma.fit(disp=-1)  # disp=-1)
            y_sim = res_arma.fittedvalues
            # plt.plot_adjust(y, color='blue')
            # plt.plot_adjust(y_sim, color='red')
            # plt.show(block=False)
            # plt.close()
            mt_arma = metrics.Metrics(y, y_sim)
            sr_met = pd.Series(index=['r2', 'ssd', 'rmse'], name='{st}_{pq}'.format(st=name_st, pq=pq_order))
            sr_met.loc['r2'] = mt_arma.r2
            sr_met.loc['ssd'] = mt_arma.ssd
            sr_met.loc['rmse'] = mt_arma.rmse
            if printval:
                print(sr_met.to_string())
            dict_res[pq_order] = res_arma
            df_met[pq_order] = sr_met
        except Exception as e:
            if printval:
                print("The ARMA{} model couldn't be performed. {}".format(pq_order, e))
            pass

    df_met.loc['r2'] = df_met.loc['r2'] == df_met.loc['r2'].max()
    df_met.loc['ssd'] = df_met.loc['ssd'] == df_met.loc['ssd'].min()
    df_met.loc['rmse'] = df_met.loc['rmse'] == df_met.loc['rmse'].min()
    ix_sel = df_met.sum()[df_met.sum() == df_met.sum().max()].index[0]  # Be careful with this line
    if printval:
        print('Selected Model: ARMA{}'.format(ix_sel))

    return dict_res[ix_sel]


def forecast_arma(date_forecast, y, t=1, df_feat=None, savefigs=False, printval=False, alpha=.05, freq='MS', std=False,
                  ret_params=False, max_ar=4, max_ma=2):
    ix_data = y.loc[:date_forecast].index
    y_train = y.loc[ix_data][:-1]

    if std:
        y_train, y_mean, y_std = hb.fn_std(x=y_train, retval=True)

    ix_fore = pd.Index(pd.date_range(date_forecast, periods=t, freq=freq, name='Date'))

    if df_feat is not None:
        ix_train = y_train.dropna().index.intersection(df_feat.dropna().index)
        y_train = y_train.loc[ix_train]
        ix_total = ix_train.union(ix_fore)
        df_feat_total = df_feat.loc[ix_total]
        df_comp = eof.fn_pca(df_feat_total)
        df_comp_train = df_comp.loc[ix_train]
        df_comp_fore = df_comp.loc[ix_fore]

    else:
        df_comp_train = None
        df_comp_fore = None

    mod_arma = core_arma(y_train, df_feat=df_comp_train, savefigs=savefigs, printval=printval, max_ar=max_ar,
                         max_ma=max_ma)
    fore_arma = mod_arma.forecast(t, df_comp_fore, alpha)

    sr_fore = pd.Series(fore_arma[0], index=ix_fore)

    if std:
        sr_fore = hb.fn_istd(sr_fore, y_mean, y_std)

    if ret_params:
        return sr_fore, mod_arma.params
    else:

        return sr_fore


if __name__ == '__main__':
    pass
