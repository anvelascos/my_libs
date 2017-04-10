import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.tsa import stattools

import hydrobasics as hb
import oai
import utilities as util
from constants import *

font = {'family': 'arial', 'weight': 'normal', 'size': 8}
plt.rc('font', **font)


class BasicData(object):
    def __init__(self, series, freq='MS'):
        """
        Basic Data for a time series.
        :param series: time series.
        :return:
        """
        self.freq = freq
        sr_data = series.dropna()
        self.start = sr_data.index[0]
        self.end = sr_data.index[-1]

        if freq == 'D':
            ix_data = hb.rem0229(pd.date_range(start=self.start, end=self.end, freq=freq))
            total = float(len(ix_data))
            sr_data = sr_data.loc[ix_data].dropna()

        elif freq == 'MS':
            total = float(len(pd.date_range(start=self.start, end=self.end, freq=freq)))
            self.srmean = hb.fn_srmean(series).loc[self.start:self.end]
            self.srstd = hb.fn_srstd(series).loc[self.start:self.end]
            self.series_ds = hb.fn_remseason(series, kind='Centralise').loc[self.start:self.end]

        else:
            total = float(len(pd.date_range(start=self.start, end=self.end, freq=freq)))

        self.ntotal = total
        n_data = float(len(sr_data))
        self.ndata = n_data
        self.gaps = 100. * (1 - n_data / total)


class BasicStatistics(object):
    def __init__(self, series):
        """
        Basic Statistics for a time series.
        :param series: time series.
        :return:
        """
        self.mean = float(series.mean())
        self.std = float(series.std())
        self.max = float(series.max())
        self.min = float(series.min())

        if self.mean == 0:
            self.cv = None

        else:
            self.cv = float(self.std / self.mean)

        self.cs = float(series.skew())


class ACFunction(object):
    def __init__(self, series, remseason=False, kind='Standardise', retval=False, name='Series', unbiased=False,
                 nlags=40, qstat=False, fft=True, alpha=None, freq='MS', fix_freq=None):
        """
        Characterisation of a data set as a time series. It is based on stattools.acf. The confidence interval is
        based on the formula disseminated for Proff. Efrain Dominguez.
        :param series:
        :param remseason:
        :param kind:
        :param retval:
        :param name:
        :param unbiased:
        :param nlags:
        :param qstat:
        :param fft:
        :param alpha:
        :param freq:
        :param fix_freq:
        """

        if freq == 'MS':
            div = 12

        elif freq == 'D':
            div = 365

        else:
            div = 12

        self.remseason = remseason
        self.name = name
        self.nlags = nlags

        if self.remseason:

            if retval:
                if kind == 'Standardise':
                    sr_series, sr_mean, sr_std = hb.fn_remseason(series, kind=kind, retval=retval)
                    self.series = sr_series.loc[series.index]
                    self.mean = sr_mean
                    self.std = sr_std

                elif kind == 'Normalise' or kind == 'Centralise':
                    sr_series, sr_mean = hb.fn_remseason(series, kind=kind, retval=retval)
                    self.series = sr_series
                    self.mean = sr_mean

                else:
                    sr_series, sr_mean, sr_std = hb.fn_remseason(series, kind=kind, retval=retval)
                    self.series = sr_series.loc[series.index]
                    self.mean = sr_mean
                    self.std = sr_std

            else:
                sr_series = hb.fn_remseason(series, kind=kind, retval=retval)
                self.series = sr_series.loc[series.index]

        else:
            self.series = series

        if not pd.isnull(self.series).any():
            acf, confint = stattools.acf(self.series, unbiased=unbiased, nlags=nlags, qstat=qstat, fft=fft, alpha=alpha)
            self.acf = pd.Series(acf, index=pd.Index(range(nlags + 1), name="Lags"))
            confint = np.ones(nlags + 1)
            confint[0] = 0.

            if fix_freq is not None:
                n_years = float(fix_freq)
                confint[1:] = hb.fn_rteo(np.ones(nlags) * n_years)

            else:
                confint[1:] = hb.fn_rteo([(len(self.series) - 1 - x) / div - 2 for x in range(nlags)])

            confint = np.array(zip(- confint, confint))
            self.confint = pd.DataFrame(confint, index=pd.Index(range(nlags + 1), name="Lags"))

        else:
            print("Time series {} has nan values, the ACF couldn't be performed.".format(name))
            self.acf = None
            self.confint = None

    def plot_acf(self, savefig=False, namefig=None, par=None, step=10):
        # TODO: Check plot.

        if self.remseason:
            title_sta = '{}-DE'.format(self.name)

        else:
            title_sta = '{}'.format(self.name)

        fig, ax = hb.gx_acf(acf_x=self.acf, confint=self.confint[1])
        ax.set_title('Funcion de Autocorrelacion {} ({})'.format(par, title_sta))
        ax.set_ylabel('Correlacion')
        ax.set_xlabel(r'$\tau$')

        if self.nlags > 50:
            major_ticks = np.arange(0, self.nlags + step, step)
            minor_ticks = np.arange(0, self.nlags + step / 5, step / 5)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticklabels(major_ticks)
            ax.grid(which='minor', alpha=.2)
            ax.grid(which='major', alpha=.5)

        plt.tight_layout()

        if savefig:
            if namefig is None:
                namefig = '{}_acf'.format(self.name)

            namefig = util.adj_name(namefig)
            plt.savefig(namefig)

        plt.close()


class FitPDF(object):
    # TODO: Revisar funciones que el valor esperado da loco.
    # Revisar que este entre los percentiles 25 y 75, si no, se debe poner el valor medio.
    # TODO: Grafica de isopercentiles que no muestre por debajo de cero
    # No pueden dar valores por debajo de cero, si es menor a cero, se debe poner cero.

    def __init__(self, gr_data=None, percentil=None, multiprocessing=False, parameter=None, name=None):
        """
        This class has the principal parameters of the best PDF for a data set.
        :param gr_data: data set grouped.
        :param multiprocessing: Multiprocess mode.
        :return:
        """

        if isinstance(gr_data, pd.Series):
            gr_data = pd.DataFrame(gr_data)

        ix_fit = pd.Index(range(1, len(gr_data.columns) + 1), name='period')
        sr_fit = pd.Series(index=ix_fit, dtype=str)
        sr_mare = pd.Series(index=ix_fit)
        dict_pars = {}
        sr_ndata = pd.Series(index=ix_fit)
        sr_mean = pd.Series(index=ix_fit)
        sr_variance = pd.Series(index=ix_fit)

        if percentil is None:
            percentil = pd.Index([.05, .25, .5, .75, .95], name='Percentile')

        df_isopercentil = pd.DataFrame(index=percentil, columns=ix_fit)
        dict_all_fitted = {}
        dict_all_pars = {}

        for period in gr_data.columns:
            sr_data = gr_data[period].dropna()
            sr_ndata[period] = len(sr_data)
            df_fit, dic_pars_fit = hb.fn_fitpdf(sr_data, multiprocessing=multiprocessing)

            if len(df_fit[df_fit['kst']]) > 0:
                fdist = df_fit.iloc[0].name
                sr_fit[period] = fdist
                sr_mare[period] = df_fit.iloc[0]['mare']
                dict_pars.update({period: dic_pars_fit[fdist]})
                cur_dist = getattr(ss, fdist)
                pars = cur_dist.fit(sr_data)
                sr_mean[period], sr_variance[period] = cur_dist.stats(*pars, moments='mv')
                df_isopercentil[period] = cur_dist.isf(percentil, *pars)
                # sr_dist = cur_dist.cdf(sr_data, *pars)

            else:
                sr_fit[period] = None
                sr_mare[period] = None
                dict_pars.update({period: None})

            dict_all_fitted[period] = df_fit
            dict_all_pars[period] = dic_pars_fit

        self.data = gr_data
        self.pdf = sr_fit
        self.pars = dict_pars
        self.mare = sr_mare
        self.runstest = hb.fn_runstest(gr_data)
        self.ndata = sr_ndata
        self.mean = sr_mean
        self.variance = sr_variance
        self.isopercentil = df_isopercentil
        self.allfitted = dict_all_fitted
        self.allparams = dict_all_pars
        self.name = name

        if parameter is None:
            self.parameter = None
            self.units = "Sin dimensiones"

        else:
            self.parameter = parameter
            self.units = dict_units[parameter[:2]]

    def plot_adjust(self, savefig=False, namefig=None):
        """
        This function plots twelve fitted PDF of a data set.
        :param savefig:
        :param namefig:
        :return:
        """
        for period in self.data.columns:
            data_obs = self.data[period].dropna().sort_values().values
            xe = 1. - hb.fn_theoreticalprobability(len(data_obs))
            rows = 3
            columns = 4
            fig, axes = plt.subplots(rows, columns, sharex=True, sharey=True, figsize=(22, 16))
            best = self.pdf[period]
            all_dists = self.allfitted[period].index.sort_values()
            all_pars = self.allparams[period]

            for i in range(rows):
                for j in range(columns):
                    k = i * (rows + 1) + j
                    fdist = all_dists[k]
                    pars = all_pars[fdist]
                    xt = 1. - getattr(ss, fdist).cdf(data_obs, *pars)

                    if fdist == best:
                        axes[i, j].plot(data_obs, xe, '.k', data_obs, xt, '-r', linewidth=3.0)

                    else:
                        axes[i, j].plot(data_obs, xe, '.k', data_obs, xt, '-k', color='0.40')

                    error = 100 * np.abs(xt - xe) / xe
                    emean = "$\epsilon_{{mean}}={emean:.2f}$%".format(emean=error.mean())
                    emax = "$\epsilon_{{max}}={emax:.2f}$%".format(emax=error.max())
                    title = "{}, {}, {}".format(fdist, emean, emax)
                    axes[i, j].set_title(title, size=10)
                    axes[i, j].legend(['Pe', fdist], fontsize='medium', loc='best')

            fig.text(0.5, 0.0, 'X', ha='center')
            fig.text(0.0, 0.5, r'$F(x)$', va='center', rotation='vertical')
            suptitle = "Ajuste de Funciones de Distribucion de Probabilidad " \
                       "(Par: {}, Serie: {}, Mes: {})".format(self.parameter, self.name, dict_months[period])
            fig.suptitle(suptitle, size=14)
            plt.tight_layout(pad=5)

            if savefig:
                if namefig is None:
                    namefig = "{}_cdffitted".format(self.name)

                plt.savefig("{}_{:02}".format(namefig, period))

            else:
                plt.show()

            plt.close()

    def plot_best(self, savefig=False, namefig=None):
        """
        This function plots the adjusted PDFs, one for each month of the year.
        :param savefig:
        :param namefig:
        :return:
        """

        if len(self.data.columns) <= 13:
            dict_plots = {i: [int(i - 1) / 4, (i - 1) % 4] for i in range(1, 13)}
            fig, arrax = plt.subplots(nrows=3, ncols=4, sharey=True)
            fig.suptitle('Funciones de densidad de probabilidad acumulada'
                         ' de mejor ajuste para la Serie {}'.format(self.name), fontsize='large')
            fig.text(.5, .03, '{} [{}]'.format(self.parameter, self.units), ha='center', va='center', fontsize='medium')
            fig.text(.03, .5, 'Probabilidad de Excedencia', ha='center', va='center', rotation='vertical',
                     fontsize='medium')

            periods = range(1, 13)
            for period in periods:
                data_obs = self.data[period].dropna().sort_values().values
                xe = 1. - hb.fn_theoreticalprobability(len(data_obs))
                best_cdf = self.pdf[period]
                pars = self.pars[period]
                row = dict_plots[period][0]
                col = dict_plots[period][1]

                if best_cdf is None:
                    arrax[row, col].plot(data_obs, xe, '.k', linewidth=1., markersize=3.)
                    arrax[row, col].legend(['Observados'], fontsize='x-small', loc='best')

                else:
                    xt = 1. - getattr(ss, best_cdf).cdf(data_obs, *pars)
                    arrax[row, col].plot(data_obs, xe, '.k', data_obs, xt, '-r', linewidth=1., markersize=3.)
                    arrax[row, col].legend(['Observados', best_cdf], fontsize='small', loc='best')

                arrax[row, col].set_title(dict_months[period], fontsize='medium')
                arrax[row, col].tick_params(axis='both', which='major', labelsize='x-small')
                arrax[row, col].grid(False)

            plt.tight_layout(rect=[0.04, 0.04, 1., 0.97])

            if savefig:
                if namefig is None:
                    namefig = "{}_cdffitted".format(self.name)

                plt.savefig(namefig, dpi=300)

            else:
                plt.show()

            plt.close()

    def rndchar_plot(self, savefig=False, namefig=None, **kwargs):
        """
        Plots random charaterise for monthly data.
        :param savefig:
        :param namefig:
        :return:
        """
        if len(self.data.columns) <= 13:
            axis_x = range(1, 13)
            fig = self.mean[axis_x].plot(color='black', linewidth=2., label='1er Momento')
            label_iso = ['PE {}%'.format(int(i * 100)) for i in self.isopercentil.index]
            len_iso = len(self.isopercentil.index) - 1
            isop_plot = self.isopercentil.T
            isop_plot.columns = label_iso
            isop_plot.plot(linewidth=.5, ax=fig)
            plt.legend(ncol=2, prop={'size': 8}, loc='best')

            try:
                for isoper in range(len_iso / 2):
                    fig.fill_between(self.mean.index, self.isopercentil.iloc[isoper],
                                     self.isopercentil.iloc[len_iso - isoper], color='gray', alpha=.15)

            except Exception, e:
                print("Theoretical characterisation plot for Series {} couldn't be plotted".format(self.name, e))

            fig.set_xticks(axis_x)
            fig.set_xticklabels([dict_months[i][:3] for i in dict_months])
            fig.set_xlabel('Meses')
            fig.set_title('Caracterizacion Teorica Serie {}'.format(self.name))
            fig.set_ylabel('{} [{}]'.format(self.parameter, self.units))
            fig.grid(True, which='major')
            fig.grid(True, which='minor')

            plt.tight_layout()

            if savefig:
                if namefig is None:
                    namefig = "{}_theochar".format(self.name)

                plt.savefig(namefig, **kwargs)

            else:
                plt.show()

            plt.close()

    def box_plot(self, savefig=False, namefig=None):
        """
        Plots Box_Plot diagram for monthly data.
        :param savefig:
        :param namefig:
        :return:
        """
        if len(self.data.columns) <= 13:
            fig = self.data.T.plot(color='gray', alpha=.05, linewidth=3.)
            # desv_above = (self.mean + 2. * (self.variance ** .5))
            # desv_below = (self.mean - 2. * (self.variance ** .5))
            # desv_above.plot(color='gray', linewidth=.5, ax=fig)
            # desv_below.plot(color='gray', linewidth=.5, ax=fig)
            # fig.fill_between(self.mean.index, desv_below, desv_above, color='gray', alpha=.25)
            axis_x = range(1, 13)
            bp = self.data[axis_x].boxplot(ax=fig, widths=.25, return_type='dict')
            plt.setp(bp['boxes'], color='black')
            plt.setp(bp['whiskers'], color='gray')
            plt.setp(bp['fliers'], color='black', marker='x')
            plt.setp(bp['medians'], color='black', linewidth=1.)

            fig.set_xticks(axis_x)
            fig.set_xticklabels([dict_months[i][:3] for i in dict_months])
            fig.set_xlabel('Meses')
            fig.set_title('Diagrama de caja y bigote Serie {}'.format(self.name))
            fig.set_ylabel('{} [{}]'.format(self.parameter, self.units))
            fig.grid(True, which='major')
            fig.grid(True, which='minor')

            plt.legend().set_visible(False)
            plt.tight_layout()

            if savefig:
                if namefig is None:
                    namefig = "{}_boxplot".format(self.name)

                plt.savefig(namefig)

            else:
                plt.show()

            plt.close()


class DurationCurve(object):
    def __init__(self, sr_data=None, name=None, par=None, freq=None):
        """
        This class calculates the duration curve of a series.
        :param sr_data:
        :param name:
        :param par:
        """

        sr_sta_dropna = sr_data.dropna()
        ix_dc = np.arange(1., len(sr_sta_dropna) + 1) / (len(sr_sta_dropna) + 1)
        sr_dc = pd.Series(sr_sta_dropna.sort_values(ascending=False).values, index=ix_dc, name="{}".format(name))
        self.durationcurve = sr_dc

        if par is not None:
            self.parameter = par[:-2]
            self.units = dict_units[par[:2]]

        self.freq = freq

    def plot(self, savefig=False, namefig=None):
        """
        Plot the duration curve.
        :param self:
        :param savefig:
        :param namefig:
        :return:
        """
        # 1. Plot the time series
        fig = self.durationcurve.plot(color='black')
        fig.set_title('Curva de Duracion ({})'.format(self.durationcurve.name))
        fig.set_ylabel('{} [{}]'.format(self.parameter, self.units))
        fig.set_xticks(np.arange(0, 1, .1))
        fig.set_xticklabels(range(0, 100, 10))
        # fig.axvline(.1, color='green')
        # fig.axvline(.9, color='green')
        fig.axvspan(0., .1, alpha=.5, color='gray')
        fig.axvspan(.1, .9, alpha=.25, color='gray')
        fig.axvspan(.9, 1., alpha=.5, color='gray')
        fig.set_xlabel('Probabilidad de Excedencia [%]')
        fig.grid(True, which='major')
        fig.grid(True, which='minor')
        plt.tight_layout()

        if savefig:

            if namefig is None:
                namefig = "{}_durationcurve".format(self.durationcurve.name)

            plt.savefig(namefig)

        else:
            plt.show()

        plt.close()


class MannKendall(object):
    def __init__(self, mg_data=None, name=None, alpha=.05):
        """

        :param mg_data:
        :param name:
        :param alpha:
        """
        self.mannkendal = hb.mk_groups(station=name, mg_data=mg_data, alpha=alpha)


class TimeSeries(object):
    def __init__(self, sr_data, name=None, par=None):
        """
        This class groups the common properties and methods of a time series and its characterisation.
        The frequency is inferred based on data index.
        :param sr_data:
        :param name:
        :param par: parameter
        :return:
        """

        freq = pd.infer_freq(sr_data.index)
        self.freq = freq

        # 0. Basic Data
        self.basicdata = BasicData(sr_data, freq=freq)
        ix_series = pd.date_range(start=self.basicdata.start, end=self.basicdata.end, freq=freq, name='Date')
        self.series = sr_data[ix_series]
        self.cdi = hb.cdi(self.series)

        if name is None:
            self.name = str(sr_data.name).translate(None, '.')
        else:
            self.name = str(name).translate(None, '.')

        if par is not None:
            self.parameter = par[:2]
            self.units = dict_units[self.parameter]

        else:
            self.parameter = None
            self.units = None

        self.basicstatistics = BasicStatistics(self.series)
        self.durationcurve = DurationCurve(self.series, par=par, freq=freq, name=self.name)

        if freq == 'MS':
            type_data = par[-1]

            if type_data == '4' or type_data == '5':
                kind_summary = 'sum'

            elif type_data == '2' or type_data == '9':
                kind_summary = 'max'

            elif type_data == '3' or type_data == '8':
                kind_summary = 'min'

            else:
                kind_summary = 'mean'

            self.monthlygroups = hb.fn_sr2mg(self.series, col_sum=True, kind_summary=kind_summary)
            self.trends = MannKendall(self.monthlygroups, name=self.name)

        elif freq == 'D':
            self.dailygroups = hb.fn_sr2dg(self.series)

        self.randomchar = None
        self.timeserieschar = None

    def plot(self, savefig=False, namefig=None, plot_ds=False, **kwargs):
        """
        Plots the time series
        :param savefig: save figure.
        :type savefig: bool
        :param namefig: figure's name.
        :type namefig: str
        :param plot_ds: plot deseasonalised data.
        :type plot_ds: bool
        :return:
        """
        if plot_ds and self.freq == 'MS':
            fig, arrax = plt.subplots(nrows=2, ncols=1, sharex=True)
            self.series.plot(ax=arrax[0], color='black', linewidth=1.)
            arrax[0].set_title('Serie de tiempo ({})'.format(self.name))
            arrax[0].set_ylabel('{} [{}]'.format(self.parameter, self.units))
            arrax[0].set_xlabel('Tiempo [{}]'.format(dict_times[self.basicdata.freq]))
            arrax[0].grid(True, which='major')
            arrax[0].grid(True, which='minor')
            
            self.basicdata.series_ds.plot(ax=arrax[1], color='black', linewidth=1.)
            arrax[1].set_title('Serie de tiempo desestacionalizada')
            arrax[1].set_ylabel('{} [{}]'.format(self.parameter, self.units))
            arrax[1].set_xlabel('Tiempo [{}]'.format(dict_times[self.basicdata.freq]))
            arrax[1].grid(True, which='major')
            arrax[1].grid(True, which='minor')

        else:
            fig = self.series.plot(color='black', linewidth=1.)
            fig.set_title('Serie de tiempo ({})'.format(self.name))
            fig.set_ylabel('{} [{}]'.format(self.parameter, self.units))
            fig.set_xlabel('Tiempo [{}]'.format(dict_times[self.basicdata.freq]))
            fig.grid(True, which='major')
            fig.grid(True, which='minor')

        plt.tight_layout()

        if savefig:
            if namefig is None:
                namefig = str(self.name) + '_timeseries'

            plt.savefig(namefig, **kwargs)

        else:
            plt.show()

        plt.close()

    def rndchar(self, multiprocessing=False, percentiles=None):
        """
        Characterises a data series as a random process.
        :param multiprocessing:
        :param percentiles:
        :return:
        """
        if self.freq == 'MS':
            self.randomchar = FitPDF(self.monthlygroups, percentil=percentiles, multiprocessing=multiprocessing,
                                     parameter=self.parameter, name=self.name)

        elif self.freq == 'D':
            self.randomchar = FitPDF(self.dailygroups, percentil=percentiles, multiprocessing=multiprocessing,
                                     parameter=self.parameter, name=self.name)

        else:
            self.randomchar = None

    def tschar(self, remseason=False, kind=None, retval=False, savefig=False, namefig=None, unbiased=False, nlags=40,
               qstat=False, fft=True, alpha=.05, fix_freq=None):
        """
        Characterises a data series as a time series. It is based on stattools.acf.
        :param remseason: remove seasonality.
        :param kind: Method for removing seasonality ('Standardise', 'Normalise', 'Centralise')
        :param retval: Return complementary seasonality data (Mean and Std if apply).
        :param savefig: save figure.
        :param namefig: figure's name.
        :param unbiased:
        :param nlags:
        :param qstat:
        :param fft:
        :param alpha:
        :param fix_freq: Fix length series for correlation analysis.
        :return:
        """
        self.timeserieschar = ACFunction(self.series, remseason=remseason, kind=kind, retval=retval, name=self.name,
                                         unbiased=unbiased, nlags=nlags, qstat=qstat, fft=fft, alpha=alpha,
                                         freq=self.freq, fix_freq=fix_freq)

    def cdi_plot(self, savefig=False, namefig=None, plot_oni=False, **kwargs):
        """
        This function plots the Integrated Differences Curve (CDI).
        :param savefig:
        :param namefig:
        :param plot_oni:
        :return:
        """
        diff = self.cdi.diff()

        if plot_oni and self.freq == 'MS':
            fig, arrax = plt.subplots(nrows=3, ncols=1, sharex=True)

            try:
                oni = util.load_obj('oai_oni', 'objs')

            except IOError:
                oni = oai.get_oai(oai_update=['oni'])
                util.save_obj(oni, 'oai_oni', 'objs')

            oni = oni.loc[self.cdi.index]
            oni.plot(ax=arrax[2], color='k', linewidth=.6, zorder=20)
            arrax[2].axhline(color='k')
            arrax[2].set_ylabel('ONI', fontsize=7)
            arrax[2].set_xlabel(r'Fecha', fontsize=7)
            arrax[2].set_title('Oceanic Nino Index (ONI)'.format(self.name), fontsize=8)
            arrax[2].grid(True, which='major')
            arrax[2].grid(True, which='minor')
            arrax[2].legend().set_visible(False)
            [arrax[2].axhspan(i, 2.5, alpha=.15, color='red') for i in np.arange(.5, 2.5, .5)]
            [arrax[2].axhspan(-i, -2., alpha=.15, color='blue') for i in np.arange(.5, 2.5, .5)]
            arrax[2].set_ylim(-2, 2.5)
            arrax[2].tick_params(axis='both', which='major', labelsize=6)

        else:
            fig, arrax = plt.subplots(nrows=2, ncols=1, sharex=True)

        diff.plot(style='k', ax=arrax[0], linewidth=.3)
        arrax[0].fill_between(self.cdi.index, diff, where=diff >= 0, color='blue', alpha=.6)
        arrax[0].fill_between(self.cdi.index, diff, where=diff <= 0, color='red', alpha=.6)
        arrax[0].axhline(color='k')
        arrax[0].set_ylabel(r'$Diferencias$', fontsize=7)
        arrax[0].grid(True, which='major')
        arrax[0].grid(True, which='minor')
        arrax[0].set_title('Curva de diferencias Serie {}'.format(self.name), fontsize=8)
        arrax[0].tick_params(axis='both', which='major', labelsize=6)

        self.cdi.plot(style='-k', linewidth=.6, ax=arrax[1])
        arrax[1].axhline(color='k', linewidth=1.)
        arrax[1].set_ylabel(r'$\xi$', fontsize=7)
        arrax[1].grid(True, which='major')
        arrax[1].grid(True, which='minor')
        arrax[1].set_title('Curva de diferencias integradas Serie {}'.format(self.name), fontsize=8)
        arrax[1].tick_params(axis='both', which='major', labelsize=6)

        plt.tight_layout()

        if savefig:
            if namefig is None:
                namefig = str(self.name) + '_cdi'

            plt.savefig(namefig, **kwargs)

        else:
            plt.show()

        plt.close()

    def smc_plot(self, savefig=False, namefig=None, **kwargs):
        """
        This function plots the Simple Mass Curve.
        :param savefig: save figure.
        :param namefig: figure's name.
        :return:
        """
        fig = self.series.cumsum().plot(style='-k', linewidth=2.)
        # fig.axhline(color='gray', linewidth=1., zorder=1)
        fig.set_title('Curva de masa simple ({})'.format(self.name))
        fig.set_ylabel('{} [{}]'.format(self.parameter, self.units))
        fig.set_xlabel('Tiempo [{}]'.format(dict_times[self.freq]))
        fig.grid(True, which='major')
        fig.grid(True, which='minor')
        plt.tight_layout()

        if savefig:
            if namefig is None:
                namefig = str(self.name) + '_smc'

            plt.savefig(namefig, **kwargs)

        else:
            plt.show()

        plt.close()


if __name__ == '__main__':
    pass
