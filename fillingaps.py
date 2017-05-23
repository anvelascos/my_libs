import hydrobasics as hb
import characterise as char
import clao as clao
import pandas as pd
import statsmodels.api as sm
import utilities as util
import multiprocessing as mp
import functools as ft


def fn_fillin(df_data_gaps, df_ranges, txt_output, negmeth='slinear'):
    new_line = '##############################################################################\n'
    col_data = df_data_gaps.columns
    ix_data = df_data_gaps.index
    df_mean_sea = hb.fn_seasonmean(df_data_gaps)
    df_std_sea = hb.fn_seasonstd(df_data_gaps)
    rem_col = col_data[df_std_sea[df_std_sea == 0].count() > 0]
    col_data = col_data.drop(rem_col)
    df_std_sea.drop(rem_col, axis=1, inplace=True)
    df_data_gaps.drop(rem_col, axis=1, inplace=True)
    df_data_gaps_std = pd.DataFrame(index=ix_data, columns=col_data, dtype=float)
    col_gaps = df_ranges.index
    df_fillin = pd.DataFrame(index=df_data_gaps.index, columns=col_gaps, dtype=float)

    for month in range(1, 13):
        ix_month = ix_data[ix_data.month == month]
        df_data_gaps_std.loc[ix_month] = (df_data_gaps.loc[ix_month] - df_mean_sea.loc[month]) / df_std_sea.loc[month]

    for station_gap in col_gaps:
        # print header_line
        ts_station_gap = char.TimeSeries(df_data_gaps_std[station_gap])
        sr_station = ts_station_gap.series
        sta_start = df_ranges.loc[station_gap, 'Start']
        sta_end = df_ranges.loc[station_gap, 'End']
        ix_fillin = pd.date_range(sta_start, sta_end, freq='MS', name='Date')
        ix_data_fill = ix_fillin[sr_station.loc[ix_fillin].isnull()]
        months_gaps = pd.unique(ix_data_fill.month)
        header_line = '\n' + 2 * new_line + 'Station: ' + station_gap + '\n' + 'Range:   ' +\
                      sta_start.strftime('%Y-%m') + ' - ' + sta_end.strftime('%Y-%m') + '\n' + 2 * new_line
        txt_output.write(header_line)

        for month in months_gaps:
            # ix_month_sta = sr_station[sr_station.index.month == month].index
            sr_gaps_month = sr_station[sr_station.index.month == month].dropna()

            # query_1: Stations with data when gaps are present.
            ix_q1 = ix_data_fill[ix_data_fill.month == month]
            col_q1 = df_data_gaps_std.loc[ix_q1].dropna(axis=1).columns
            df_data_q1 = df_data_gaps_std.loc[sr_gaps_month.index, col_q1]

            # query_2: Stations with more than 90% of fullness in the date axis.
            min_data_2 = len(df_data_q1.index) * .9
            sr_count_2 = df_data_q1.count(axis=0)
            df_data_q2 = df_data_q1[sr_count_2[sr_count_2 > min_data_2].index]

            # query_3: Months with more than 95% of fullness in the station axis.
            min_data_3 = len(df_data_q2.columns) * .95
            sr_count_3 = df_data_q2.count(axis=1)
            df_data_q3 = df_data_q2.loc[sr_count_3[sr_count_3 > min_data_3].index]

            # query_4: Stations with significant correlation
            df_data_q4 = df_data_q3.dropna(axis=1)
            ix_pred = df_data_q4.index

            sr_corr = df_data_q4.corrwith(sr_station.loc[ix_pred]).abs().sort_values(ascending=False, inplace=False)
            rmin = hb.fn_rteo(len(sr_station.loc[ix_pred]))
            sta_sel = sr_corr[sr_corr >= rmin].index
            month_line = '\nMonth: {month:02}, Significance Correlation: {corr:.2f}\n'.format(month=month, corr=rmin)
            txt_output.write(month_line)
            # print month_line

            df = len(ix_pred) - len(sta_sel)  # Degrees of freedom
            if df < 2:
                sta_sel = sta_sel[:len(ix_pred) - 2]

            ix_total = ix_pred.append(ix_q1).sort_values()
            df_feat = df_data_gaps_std.loc[ix_total, sta_sel]
            x_total = sm.add_constant(df_feat, prepend=False)
            x_train = x_total.loc[ix_pred]
            y_train = sr_station.loc[ix_pred]

            pars, summary = clao.fn_regrem(x_train, y_train, retsum=True, max_cn=1000)

            if pars is not None:
                x_fill = x_total.loc[ix_q1, pars.index]

                if 'const' in pars.index:
                    x_fill.loc['const'] = 1
                y_fill = (x_fill * pars).sum(axis=1)

            else:
                y_fill = None

            df_fillin.loc[ix_q1, station_gap] = y_fill
            sum_line = '\n' + str(summary) + '\n'
            txt_output.write(sum_line)

    df_output = pd.DataFrame(index=df_data_gaps.index, columns=col_gaps, dtype=float)
    for month in range(1, 13):
        ix_month = ix_data[ix_data.month == month]
        df_output.loc[ix_month] = df_fillin.loc[ix_month] * df_std_sea.loc[month, col_gaps] + df_mean_sea.loc[month,
                                                                                                              col_gaps]

    df_output[df_output.isnull()] = df_data_gaps[col_gaps]
    df_output[df_output < 0] = None

    txt_output.write('\n\n{}{}\nSTATIONS WITH NEGATIVES VALUES FILLED IN\n'.format(new_line, new_line))

    for station_gap in col_gaps:
        sta_start = df_ranges.loc[station_gap, 'Start']
        sta_end = df_ranges.loc[station_gap, 'End']
        ix_fillin = pd.date_range(sta_start, sta_end, freq='MS')
        df_output.loc[~df_output.index.isin(ix_fillin), station_gap] = None
        sr_station = df_output.loc[ix_fillin, station_gap]
        if sr_station.isnull().any():
            ix_nan = sr_station[sr_station.isnull()].index
            txt_output.write('\nStation: {}\nDates: {}\n'.format(station_gap, list(ix_nan.astype(str))))
            df_output.loc[ix_fillin, station_gap] = sr_station.interpolate(method=negmeth,
                                                                           limit_direction='both').ffill().bfill()

    return df_output


def core_group_data(day, years, stations, df_data_gaps):
    print(day)
    df_group = pd.DataFrame(index=years, columns=stations)

    for col in stations:
        sr_data = df_data_gaps[col]
        dg_data = hb.fn_sr2dg(sr_data)
        df_group[col] = dg_data[day]

    util.save_obj(df_group, 'FPK_group_{:03}'.format(day), 'objs')


def fn_fillin_daily(df_data_gaps, multiprocess=False):
    days = range(1, 366)
    years = range(df_data_gaps.index.year.min(), df_data_gaps.index.year.max() + 1)
    stations = df_data_gaps.columns

    partial = ft.partial(core_group_data, years=years, stations=stations, df_data_gaps=df_data_gaps)

    if multiprocess:
        pool = mp.Pool()
        pool.map(partial, days)
        pool.close()
        pool.join()

    else:
        map(partial, days)


def fill_example():
    """
    Modificaciones

    05-12-2014
        - Se repetian los resultados de la ultima regresion en el archivo de resultados.
        - Se incorpora complementar datos hacia atras de la fecha inicial (bandera atras).
        - Se cambia el nombre del indice de "Fecha" a "Date"
    """

    #La informacion se debe organizar de acuerdo al archivo de muestra de precipitacion.
    proyecto = 'TNC'
    parametro = 'DATA'
    atras = False

    #Archivo de lectura donde esta la informacion con los datos faltantes
    xls_input = pd.ExcelFile(proyecto + '_' + parametro + '_GAPS.xlsx')
    xls_output = pd.ExcelWriter(proyecto + '_' + parametro + '_FILLEDIN.xlsx')
    txt_output = open(proyecto + '_' + parametro + '_SUMMARY.txt', 'w')

    str_header = 'Parameter: ' + proyecto + '_' + parametro + '\n'
    # print str_header
    txt_output.write(str_header)

    #Rangos de fecha de cada una de las estaciones
    df_ranges = xls_input.parse('RANGES', index_col='Station')
    df_data_gaps = xls_input.parse(parametro, index_col='Date')

    # fill_v0(df_data_gaps, df_ranges, .2)
    df_fillin = fn_fillin(df_data_gaps, df_ranges, txt_output)

    df_fillin.to_excel(xls_output, parametro, merge_cells=False)
    xls_output.save()


if __name__ == '__main__':
    pass
