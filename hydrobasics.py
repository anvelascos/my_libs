import matplotlib as mpl
mpl.use('Agg')
import calendar
import functools as ft
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.stats as ss
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import ImageGrid
from statsmodels.graphics import utils

import my_libs.utilities as util
from constants import *

# plt.style.use('ggplot')
fontP = FontProperties()
fontP.set_size('small')


def fn_sr2mg(sr_ts):
    """
    This function transforms a time series into a dataframe monthly grouped.
    :param sr_ts: pandas time series to be transformed.
    :return: pandas dataframe monthly grouped.
    """
    df_data = pd.DataFrame(sr_ts)
    df_data['year'] = df_data.index.year
    df_data['month'] = df_data.index.month
    df_mg = df_data.pivot(index='year', columns='month', values=sr_ts.name)
    return df_mg


def fn_mg2sr(mg_input, name='Series'):
    """
    This function transforms a dataframe monthly grouped into a time series.
    :param mg_input: pandas dataframe monthly grouped to be transformed.
    :param name: output time series name.
    :return: pandas time series.
    """

    start_date = pd.datetime(mg_input.index.min(), 1, 1)
    end_date = pd.datetime(mg_input.index.max(), 12, 1)
    sr_try = mg_input.unstack()
    sr_try.index.levels[0].name = 'month'
    df_try = sr_try.reset_index()
    df_try['Date'] = pd.to_datetime(df_try['year'].astype('str') + df_try['month'].astype('str'), format="%Y%m")
    df_try.set_index(df_try['Date'], inplace=True)
    df_try.drop(['year', 'month', 'Date'], axis=1, inplace=True)
    df_try.sort_index(inplace=True)
    index_output = pd.DatetimeIndex(freq='MS', start=start_date, end=end_date, name='Date')
    sr_output = df_try.loc[index_output, 0]
    sr_output.name = name
    return sr_output


def rem0229(ix_input):
    """
    This fucntion removes february 29th from a daily index.
    :param ix_input:
    :return:
    """
    ix_output = ix_input.drop(ix_input[(ix_input.month == 2) & (ix_input.day == 29)])
    return ix_output


def fn_sr2dg(sr_input):
    """
    This function transforms a time series into a dataframe daily grouped.
    :param sr_input:
    :return:
    """
    name = sr_input.name
    ix_input = sr_input.index
    ix_rem0229 = rem0229(ix_input)
    # sr_rem0229 = sr_input.loc[ix_rem0229]
    df_dg = pd.DataFrame(index=ix_rem0229, columns=['year', 'day', name])
    df_dg['year'] = ix_rem0229.year
    df_dg['day'] = pd.Series(ix_rem0229.strftime('%j').astype(int), index=ix_rem0229, name='jday')
    df_dg[name] = sr_input.loc[ix_rem0229]
    leap_years = list({year for year in ix_rem0229.year if calendar.isleap(year)})
    ix_correct = ix_rem0229[df_dg['year'].isin(leap_years) & (ix_rem0229.month > 2)]
    df_dg.loc[ix_correct, 'day'] = df_dg['day'] - 1
    # df_dg.to_clipboard()
    dg_output = df_dg.pivot(index='year', columns='day', values=name)

    return dg_output


def fn_dg2sr(dg_input, name='Series'):
    """
    This function transforms a dataframe monthly grouped into a time series.
    :param dg_input: pandas dataframe monthly grouped to be transformed.
    :param name: output time series name.
    :return: pandas time series.
    """
    ix_input = dg_input.index
    start_date = pd.datetime(dg_input.index.min(), 1, 1)
    end_date = pd.datetime(dg_input.index.max(), 12, 31)
    sr_try = dg_input.unstack()
    sr_try.index.levels[0].name = 'jday'
    df_try = sr_try.reset_index()
    leap_years = list({year for year in ix_input if calendar.isleap(year)})
    df_try['jday'][(df_try['year'].isin(leap_years)) & (df_try['jday'] > 59)] += 1
    df_try['Date'] = pd.to_datetime(df_try['year'].astype('str') + df_try['jday'].astype('str'), format="%Y%j")
    df_try.set_index(df_try['Date'], inplace=True)
    df_try.drop(['year', 'jday', 'Date'], axis=1, inplace=True)
    df_try.sort_index(inplace=True)
    index_output = pd.DatetimeIndex(freq='D', start=start_date, end=end_date, name='Date')
    sr_output = df_try.loc[index_output, 0]
    sr_output.name = name
    return sr_output


def gx_acf(acf_x, confint, **kwargs):
    """
    This function plots the autocorrelation function, it is based on statsmodels.graphics.tsaplots.
    :param acf_x: Autocorrelation Function
    :param confint: Confidence interval
    :param kwargs:
    :return:
    """
    fig, ax = utils.create_mpl_ax()
    lags = np.arange(len(acf_x))

    pd.Series(acf_x, index=lags).plot(kind='bar', ax=ax, alpha=1., color='gray')
    # ax.vlines(lags, [0], acf_x, **kwargs)
    ax.axhline(color='black', linewidth=2.)

    # kwargs.setdefault('marker', 'o')
    # kwargs.setdefault('markersize', 3)
    # kwargs.setdefault('linestyle', 'None')
    ax.margins(.05)
    # ax.plot_adjust(lags, acf_x, **kwargs)
    filled = ax.fill_between(lags, -confint, confint, alpha=.4, color='gray')
    fill_col = filled._facecolors[0]
    p1 = plt.Rectangle((0, 0), 1, 1, 0, fc=fill_col)
    plt.legend([p1], ['No significativa'])

    return fig, ax


def fn_std(x, retval=False):
    """
    This function standardises a time series or data frame.
    :param x: series or data frame for standardising
    :return: standardised series or data frame
    """
    x_std = (x - x.mean()) / x.std()

    if retval:
        return x_std, x.mean(), x.std()
    else:
        return x_std


def fn_istd(x_std, mean, std):
    """
    Inverse of the function standardise, in other words, this function de-standardise a time series or a data frame
    :param x_std: standardised series or data frame
    :param mean: mean of de-normalised series or data frame
    :param std: standard deviation of de-standardised series or data frame
    :return: de-standardised series or data frame
    """
    x = (x_std * std) + mean
    return x


def fn_cen(x, retval=False):
    """
    This function centralises a time series or data frame.
    :param x: series or data frame for centralising
    :return: centralised series or data frame
    """
    x_cen = x - x.mean()
    if retval:
        return x_cen, x.mean()
    else:
        return x_cen


def fn_icen(x_cen, mean):
    """
    Inverse of the function centralise, in other words, this function de-centrilises a time series or a data frame
    :param x_cen: centralised series or data frame
    :param mean: mean of de-centralised series or data frame
    :return: de-centralised series or data frame
    """
    x = x_cen + mean
    return x


def fn_nor(x, retval=False):
    """
    This function normalises a time series or data frame.
    :param x: series or data frame for normalising
    :return: normalised series or data frame
    """
    x_nor = x / x.mean()
    if retval:
        return x_nor, x.mean()
    else:
        return x_nor


def fn_inor(x_nor, mean):
    """
    Inverse of the function normalise, in other words, this function de-normalises a time series or a data frame.
    :param x_nor: normalised series or data frame
    :param mean: mean of de-normalised series or data frame
    :return: de-normalised series or data frame
    """
    x = x_nor * mean
    return x


def fn_remseason(sr_input, kind='Normalise', retval=False):
    """
    This function removes seasonality of a series by subtracting the seasonal average.
    :param sr_input: series for removing seasonality.
    :param kind: 'Normalise' (Default), 'Standardise', 'Centralise'.
    :return: series without seasonality component.
    """
    mg_input = fn_sr2mg(sr_input)
    kind = kind.lower().strip()

    if retval:
        if kind == 'standardise':
            mg_diff, sr_mean, sr_std = fn_std(mg_input, retval)
            return fn_mg2sr(mg_diff, name='{name}_de'.format(name=sr_input.name)), sr_mean, sr_std

        elif kind == 'centralise':
            mg_diff, sr_mean = fn_cen(mg_input, retval)
            return fn_mg2sr(mg_diff, name='{name}_de'.format(name=sr_input.name)), sr_mean

        elif kind == 'normalise':
            mg_diff, sr_mean = fn_nor(mg_input, retval)
            return fn_mg2sr(mg_diff, name='{name}_de'.format(name=sr_input.name)), sr_mean

        else:
            print 'Caution, method {] is not enabled. Standardise will be used for removing seasonality.'.format(kind)
            mg_diff, sr_mean, sr_std = fn_nor(mg_input, retval)
            return fn_mg2sr(mg_diff, name='{name}_de'.format(name=sr_input.name)), sr_mean, sr_std

    else:
        if kind == 'standardise':
            mg_diff = fn_std(mg_input)

        elif kind == 'centralise':
            mg_diff = fn_cen(mg_input)

        elif kind == 'normalise':
            mg_diff = fn_nor(mg_input)

        else:
            print 'Caution, the method {kind} is not enabled. Standardise will be used for removing seasonality.'.format(kind=kind)
            mg_diff = fn_nor(mg_input)

        return fn_mg2sr(mg_diff, name='{name}_nsea'.format(name=sr_input.name))


def fn_diff(sr_input, n=1):
    """
    Differentiate a Series.
    :param sr_input:
    :param n:
    :return:
    """
    if n < 1:
        return sr_input

    sr_output = sr_input.copy()
    for i in range(n):
        sr_output += -sr_output.shift(1)

    sr_output.name = '{name}_diff_{n}'.format(name=sr_input.name, n=n)
    return sr_output.dropna()


def fn_runstest(df_input, alpha=.05):
    """
    Runs Test for randomness.
    :param df_input:
    :param alpha: significance.
    :return: Boolean. True for a randomness data set, False for a non randomness data set.
    """
    n = df_input.shape[0]
    rteo = (n + 1) / 2  # Theoretic Runs
    steo = (n - 1) ** .5 / 2  # Theoretic Standard Deviation
    t = ss.t.isf(alpha, n)  # t Distribution percentile for alpha
    lim_low = rteo - t * steo
    lim_upp = rteo + t * steo  #
    remp = np.sign((np.sign(df_input - df_input.mean())).diff()).abs().sum() + 1  # Empirical Runs

    return (remp > lim_low) & (remp <= lim_upp)


def ckolmo(obs):
    """
    implementa p(lamda)
    :param obs: es el valor de lamda empirico
    :return:
    """
    # vns = [0.4000, 0.3000, 0.2000, 0.1000, 0.0500, 0.0250, 0.0100, 0.0050, 0.0010, 0.0005]
    vns = [obs]
    lvns = len(vns)
    ckd = []
    for i in range(lvns):
        ckd.append([0])
    for i in range(lvns):
        ckd[i] = (-np.log(vns[i] / 2.0) / np.log(np.e) / 2.0) ** (1 / 2.0)
    return ckd


def fn_testkolmogorov(pe, pt, alfa=0.05):
    """

    :param pe: probabilidad de excedencia empirica.
    :param pt: probabilidad de excedencia teorica.
    :param alfa: nivel de significancia.
    :return: res = 1 la hipotesis no se rechaza; res = 0 La hipotesis se rechaza
    """
    d = np.max(np.abs(pe - pt))
    lamda = d * np.sqrt(np.size(pe, axis=0))  # lamda es el lamda empirico
    # print lamda
    lamdat = ckolmo(alfa)
    # print 'lamdas=', lamdat, lamda
    if lamda <= lamdat:
        res = True
    else:
        res = False
    return res


def fn_core_adjust_cdf(fdist, sr_adjust, sr_empirical):
    """

    :param fdist:
    :param sr_adjust:
    :param sr_empirical:
    :return:
    """
    sr_results = pd.Series(index=['kst', 'mare'], name=fdist, dtype=object)
    cur_dist = getattr(ss, fdist)
    pars = cur_dist.fit(sr_adjust)
    sr_dist = cur_dist.cdf(sr_adjust, *pars)
    sr_results['kst'] = fn_testkolmogorov(sr_empirical, sr_dist, .05)
    sr_results['mare'] = np.mean(np.abs(sr_empirical - sr_dist) / sr_empirical)
    return sr_dist, sr_results, pars


def fn_fitpdf(sr_input, dist_set='basics', multiprocessing=False):
    """
    This function fits a probability density function to a data series.
    :param sr_input: series for fitting pdf.
    :param dist_set: set of theoretical functions for trying. 'all', 'basics'
    :param multiprocessing: Multiprocess mode.
    :return: PDFs that approve the Smirnov-Kolmogorov Test, params of these functions.
    """
    if dist_set == 'all':
        sel_dist = [d for d in dir(ss) if isinstance(getattr(ss, d), ss.rv_continuous)]
        rem_dist = ['beta', 'dweibull', 'erlang', 'gausshyper', 'ksone', 'levy_stable', 'betaprime', 'mielke', 'ncf',
                    'nct', 'ncx2', 'tukeylambda']  # Series to remove from distribution list.
        for remove in rem_dist:
            sel_dist.remove(remove)
    elif dist_set == 'basics':
        sel_dist = ['norm', 'lognorm', 'expon', 'gamma',  'loggamma',  'gengamma', 'gumbel_l', 'gumbel_r', 'powerlaw',
                    'genextreme', 'weibull_max', 'weibull_min']
    else:
        sel_dist = ['norm', 'lognorm', 'expon', 'gamma', 'gumbel_l', 'gumbel_r']

    cols_dist = pd.Index(['empirical'] + sel_dist, name='dist')
    index_dist = pd.Index(range(sr_input.size), name='order')
    df_dist = pd.DataFrame(index=index_dist, columns=cols_dist)
    sr_input.sort_values(ascending=True, inplace=True)
    sr_input.reset_index(drop=True, inplace=True)
    df_dist['empirical'] = np.arange(1, sr_input.size + 1) / (sr_input.size + 1.)
    cols_results = pd.Index(['kst', 'mare'])
    df_results = pd.DataFrame(index=sel_dist, columns=cols_results)
    dic_pars = {}
    # df_plot = pd.DataFrame(index=sr_input.index, columns=cols_dist)

    partial_dist = ft.partial(fn_core_adjust_cdf, sr_adjust=sr_input, sr_empirical=df_dist['empirical'])

    if multiprocessing:
        pool = mp.Pool()
        results = pool.map(partial_dist, sel_dist)
        pool.close()
        pool.join()
    else:
        results = map(partial_dist, sel_dist)

    for result in results:
        # if result[1]['kst']:
        fdist = result[1].name
        # sr_dist = pd.Series(result[0], index=sr_input.index, name=df_single.index[0])
        # df_plot.update(sr_dist)
        df_results.loc[fdist] = result[1]
        dic_pars.update({fdist: result[2]})

    df_results.dropna(inplace=True)
    df_results.sort_values(['mare'], inplace=True)
    # df_plot.dropna(axis=1, inplace=True)
    return df_results, dic_pars


def fn_lagsr(sr_input, lags, freq=None):
    lags.sort()
    ix_input = str(sr_input.name)
    ix_output = [ix_input + '_R' + str(x).zfill(2) for x in lags]
    df_output = pd.DataFrame(columns=ix_output)

    for lag in lags:
        name = ix_input + '_R' + str(lag).zfill(2)
        df_output[name] = sr_input.shift(periods=lag, freq=freq)

    return df_output


def fn_rteo(df_df, alpha=.05):
    """
    This formula calculates the critical value of Pearson's Correlation.
    :param df_df: degree freedom
    :param alpha: significance level
    :return:
    """
    tteo = ss.t.isf(alpha, df_df)
    rteo = (tteo ** 2 / (tteo ** 2 + (df_df))) ** 0.5
    return rteo


def fn_rteo_nueva(df_r, df_n, alpha=.05):
    tteo = ss.t.isf(alpha / 2, df_n - 2)
    rteo = (tteo * (1. - df_r ** 2)) / np.sqrt(df_n - 1)
    return rteo


def fn_correlation(sr_hydro, df_ioa, nlags=12, freq='MS', fixfreq=None):
    if freq == 'MS':
        div = 12
    elif freq == 'D':
        div = 365
    elif freq == 'H':
        div = 8760
    else:
        div = 12

    if nlags < 0:
        lags = pd.Index(range(1, -nlags + 1), name='lag')
    else:
        lags = pd.Index(range(-nlags, 0), name='lag')

    df_corr = pd.DataFrame(index=lags, columns=df_ioa.columns)
    df_n = pd.DataFrame(index=lags, columns=df_ioa.columns, dtype=float)
    sr_ready = sr_hydro.dropna()

    for i in lags:
        df_ready = df_ioa.shift(-i)

        for ioa in df_ioa.columns:
            sr_ioa = df_ready[ioa].copy().dropna()

            if len(sr_ioa.index.intersection(sr_hydro.index)) != 0:
                if fixfreq is not None:
                    div = len(sr_ioa.index.intersection(sr_hydro.index)) / float(fixfreq)

                df_corr.loc[i, ioa] = sr_ioa.corr(sr_ready, min_periods=120)
                df_n.loc[i, ioa] = len(sr_ioa.index.intersection(sr_hydro.index)) / div

    df_corr.dropna(how='all', axis=1, inplace=True)
    rteo = fn_rteo(df_n)
    df_signif = df_corr.abs() >= rteo[df_corr.columns]
    ix_signif = df_signif.any(axis=0)[df_signif.any(axis=0)].index
    sr_lags = pd.Series(index=ix_signif, dtype=object)

    for oai in ix_signif:
        sr_lags.loc[oai] = list(abs(df_signif[oai][df_signif[oai]].index))

    return df_corr.T, rteo.T, sr_lags


def gx_corr(df_corr, name_sr, name_df, df_rteo, savefig=True, namefig='corr_oai', plotval=True, mag=1, freq='MS',
            maxshow=None, step=6, cbar_location='right'):
    """
    This function plots the correlation matrix between a series and a dataframe.
    :param df_corr: correlation matrix
    :param name_sr: name of the first variable (Series)
    :param name_df: name of the second set of variables (Dataframe)
    :param df_rteo: correlation theoretical value
    :param savefig: save figure
    :param namefig: name figure
    :param plotval: plot_adjust values instead of mark for significance correlation
    :param mag: magnifier for texts
    :param freq: data frequency
    :param maxshow: max number of items to show in the graph. If None, it plots all items.
    :param step: step for x axis ticks.
    :param cbar_location: colorbar location.
    :return:
    """

    if not (maxshow is None) and (len(df_corr.index) > maxshow):
        ix_sel = df_corr[df_corr > df_rteo].count(axis=1).sort_values(ascending=False).iloc[:maxshow].index
        df_corr = df_corr[df_corr.index.isin(ix_sel)]
        df_rteo = df_rteo[df_rteo.index.isin(ix_sel)]

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), direction='row', axes_pad=0.05, add_all=True, label_mode='l',
                     share_all=False, cbar_location=cbar_location, cbar_mode='single', cbar_size='2%', cbar_pad='1%',
                     aspect=True)
    ax = grid[0]
    # fig.subplots_adjust(top=1.2)
    # plt.title('Correlacion cruzada entre {} y {}'.format(name_sr, name_df), fontsize=10 * mag)
    plt.suptitle('Correlacion cruzada {} vs. {}'.format(name_sr, name_df), fontsize=10 * mag)
    ax.tick_params(axis='both', direction='out', labelsize=10 * mag)
    im = ax.imshow(df_corr.values.astype(float), cmap='bwr_r', aspect='equal', interpolation='none', alpha=1, vmax=1.,
                   vmin=-1.)
    sign = np.where(np.abs(df_corr) >= df_rteo)
    cbar = ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=6 * mag)

    if plotval:
        cbar.set_label_text(label='Correlacion (Valor: Significativa)', size=6 * mag)
        for i in range(len(sign[0])):
            corr_text = '{:.2f}'.format(np.abs(df_corr.iloc[sign[0][i], sign[1][i]]))[1:]
            ax.text(sign[1][i], sign[0][i], corr_text, fontsize=6 * mag, verticalalignment='center',
                    horizontalalignment='center', color='black')
    else:
        cbar.set_label_text(label='Correlacion (+: Significativa)', size=6 * mag)
        for i in range(len(sign[0])):
            ax.text(sign[1][i], sign[0][i], '+', fontsize=6 * mag, verticalalignment='center',
                    horizontalalignment='center')

    nlags = df_corr.shape[1]
    ax.tick_params(labelsize=6 * mag)

    if nlags > 30:
        major_ticks_label = np.arange(-nlags, 0, step)
        major_ticks = np.arange(0, nlags - 1, step)
        minor_ticks = np.arange(0, nlags - 1 + step / 6, step / 6)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticklabels(major_ticks_label)
        ax.xaxis.grid(True, 'minor')
        ax.xaxis.grid(True, 'major')

    else:
        ax.set_xticks(np.arange(df_corr.shape[1]))
        ax.set_xticklabels(df_corr.columns)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)

    ax.set_yticks(np.arange(df_corr.shape[0]))
    ax.set_yticklabels(df_corr.index)
    ax.set_ylabel('Variables', size=8 * mag)
    ax.set_xlabel(r'$\tau$', size=8 * mag)
    # plt.tight_layout()

    if savefig:
        namefig = util.adj_name(namefig)
        plt.savefig(namefig, dpi=600, loc='center')
        plt.close()
    else:
        plt.show()


def cdi(df_input):
    """
    This functions calculates the curve of integrated differences.
    :param df_input:
    :return:
    """
    mean = df_input.mean()
    std = df_input.std()
    cv = mean / std
    k = ((df_input / mean) - 1) / cv
    zita = k.cumsum()

    return zita


def fn_seasonmean(df_input):
    """
    This function calculates the Multiyear Monthly Mean.
    :param df_input:
    :return:
    """
    ix_mymm = pd.Index(range(1, 13), name='Month')
    df_mymm = pd.DataFrame(index=ix_mymm, columns=df_input.columns)
    for month in ix_mymm:
        df_mymm.loc[month] = df_input[df_input.index.month == month].mean(axis=0)
    return df_mymm


def fn_seasonstd(df_input):
    """
    This function calculates the Multiyear Monthly Standar Deviation.
    :param df_input:
    :return:
    """
    ix_mymm = pd.Index(range(1, 13), name='Month')
    df_mymm = pd.DataFrame(index=ix_mymm, columns=df_input.columns)
    for month in ix_mymm:
        df_mymm.loc[month] = df_input[df_input.index.month == month].std(axis=0)
    return df_mymm


def fn_srmean(df_input):
    ix_input = df_input.index
    df_mean_sr = df_input.copy().replace()
    df_mean_sr.loc[ix_input] = None
    for date in ix_input:
        df_mean_sr.loc[date] = df_input[(ix_input.month == date.month) & (ix_input.year < date.year)].mean(axis=0)

    return df_mean_sr


def fn_srstd(df_input):
    ix_input = df_input.index
    df_std_sr = df_input.copy().replace()
    df_std_sr.loc[ix_input] = None

    for date in ix_input:
        df_std_sr.loc[date] = df_input[(ix_input.month == date.month) & (ix_input.year < date.year)].std(axis=0)

    return df_std_sr


def fn_theoreticalprobability(n_data):
    return pd.Series(np.arange(1, n_data + 1) / (n_data + 1.))


def mk_serie(sr_data, alpha=0.05):
    """
    Esta funcion realiza el test de Mann-Kendall para evaluar si existe tendencia en una serie de datos
    :param sr_data: Vector con los datos de entrada
    :param alpha: Nivel de significancia estadistica de la prueba (0.05 default)
    :return: Tendencia: Creciente, decreciente or sin tendencia, h: Verdadero or Falso, p: p value,
    z: test estadistico normalizado
    """

    # Numero de datos
    sr_data.dropna(inplace=True)
    n = len(sr_data)

    # Calculo de S
    s = np.sum([np.sign(sr_data.diff(i)).sum() for i in range(n - 1)])

    # for k in range(n - 1):
    #     for j in range(k + 1, n):
    #         s += np.sign(sr_data.iloc[j] - sr_data.iloc[k])

    # Extraccion de los valores unicos de la serie
    unique_x = np.unique(sr_data)
    g = len(unique_x)

    # Calculo de var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18

    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)

        for i in range(len(unique_x)):
            tp[i] = sum(sr_data == unique_x[i])

        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    z = 0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)

    elif s == 0:
        z = 0

    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    # Calculo del p_value
    # p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > ss.norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        tendencia = 'Decreciente'

    elif (z > 0) and h:
        tendencia = 'Creciente'

    else:
        tendencia = 'Sin tendencia'

    # m_k = pd.DataFrame(index=[station], columns=['Tendencia', 'h', 'p', 'z'])
    # m_k['Tendencia'], m_k['h'], m_k['p'], m_k['z'] = tendencia, h, p, z
    # return m_k
    return tendencia


def mk_groups(station, mg_data, alpha=0.05):
    """
    Esta funcion realiza el test de Mann-Kendall a un conjunto de datos estadisticos (grupo de datos mensuales)
    :param station: Estacion evaluada
    :param mg_data: DataFrame con los datos de entrada
    :param alpha: Nivel de significancia estadistica de la prueba (0.05 default)
    :return: Tendencia: Creciente, decreciente or sin tendencia, h: Verdadero or Falso, p: p value,
    z: test estadistico normalizado
    """

    # Conjuntos estadisticos
    groups = mg_data.columns

    # DataFrame de salida
    tendencias = pd.Series(name=station, index=groups)

    # Ciclo para aplicar el test de Mann-Kendall a los conjuntos estadisticos
    for group in groups:
        sr_data = mg_data[group]
        tendencias.loc[group] = mk_serie(sr_data=sr_data, alpha=alpha)

    # DataFrame de salida
    return tendencias


if __name__ == '__main__':
    pass
