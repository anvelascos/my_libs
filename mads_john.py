import pandas as pd
import numpy as np
import functools as ft
import multiprocessing as mp
from hydrobasics import fn_sr2mg, fn_mg2sr


def mads_monthly(sta, df_data, k):
    df_serie = pd.DataFrame(df_data[sta])
    df_serie.columns = ['data']
    df_serie['year'] = pd.to_datetime(df_serie.index).year
    df_serie['month'] = pd.to_datetime(df_serie.index).month
    # nyears = ((df_serie['Year'].max() - df_serie['Year'].min()) + 1)

    df_daily = df_serie.pivot(index='year', columns='month', values='data')
    m = np.nanmedian(df_daily, axis=0)
    abs_dev = df_daily.subtract(m).abs()
    left_mad = np.ma.median(np.ma.masked_array(abs_dev[df_daily <= m], np.isnan(abs_dev[df_daily <= m])))
    right_mad = np.ma.median(np.ma.masked_array(abs_dev[df_daily > m], np.isnan(abs_dev[df_daily > m])))
    left = np.ma.masked_array(abs_dev[df_daily <= m], np.isnan(abs_dev[df_daily <= m])) / left_mad
    right = np.ma.masked_array(abs_dev[df_daily > m], np.isnan(abs_dev[df_daily > m])) / right_mad
    df_clean = np.ma.filled(left, 0) + np.ma.filled(right, 0)
    df_clean = pd.DataFrame(data=df_clean, index=df_daily.index, columns=df_daily.columns)
    df_clean[df_clean == 0] = np.nan
    df_clean[df_daily == m] = 0
    df_clean[df_clean > k] = np.nan
    df_clean[df_clean <= k] = df_daily

    sr_clean = fn_mg2sr(df_clean)
    sr_clean.name = sta
    # p1 = df_clean.unstack()
    # p2 = pd.DataFrame(p1)
    # p2.reset_index(inplace=True)
    # p2.sort_values(['Year', 'Month'], inplace=True)
    # sr_clean = pd.Series(data=p2.iloc[:, 2].values, index=df_data.index, name=sta)
    return sr_clean


def mads_daily(sta, df_data, k):
    # print sta
    df_serie = pd.DataFrame(df_data[sta])
    df_serie.columns = ['Data']
    df_serie['Year'] = pd.to_datetime(df_serie.index).year
    nyears = ((df_serie['Year'].max() - df_serie['Year'].min()) + 1)
    df_serie['DayJulian'] = nyears * range(1, 366)
    df_daily = df_serie.pivot(index='Year', columns='DayJulian', values='Data')
    m = np.nanmedian(df_daily, axis=0)
    abs_dev = df_daily.subtract(m).abs()
    left_mad = np.ma.median(np.ma.masked_array(abs_dev[df_daily <= m], np.isnan(abs_dev[df_daily <= m])))
    right_mad = np.ma.median(np.ma.masked_array(abs_dev[df_daily > m], np.isnan(abs_dev[df_daily > m])))
    left = np.ma.masked_array(abs_dev[df_daily <= m], np.isnan(abs_dev[df_daily <= m])) / left_mad
    right = np.ma.masked_array(abs_dev[df_daily > m], np.isnan(abs_dev[df_daily > m])) / right_mad
    df_clean = np.ma.filled(left, 0) + np.ma.filled(right, 0)
    df_clean = pd.DataFrame(data=df_clean, index=df_daily.index, columns=df_daily.columns)
    df_clean[df_clean == 0] = np.nan
    df_clean[df_daily == m] = 0
    # print df_clean.median(axis=0)
    df_clean[df_clean > k] = np.nan
    df_clean[df_clean <= k] = df_daily

    p1 = df_clean.unstack()
    p2 = pd.DataFrame(p1)
    p2.reset_index(inplace=True)
    p2.sort_values(['Year', 'DayJulian'], inplace=True)
    sr_clean = pd.Series(data=p2.iloc[:, 2].values, index=df_data.index, name=sta)
    # sr_clean.plot()
    # plt.show()
    # sr_clean
    return sr_clean


def main():
    vartype = ['PT_4']
    project = 'Test'
    cutoff = 4.0
    multiprocessing = False
    for var in vartype:
        path_data = 'Series_Raw/' + project + '/' + var + '.csv'
        path_out = 'Series_Clean/' + project + '/' + var + '.csv'
        df_data = pd.read_csv(path_data, index_col='Date')
        sta = df_data.columns
        partial_info = ft.partial(mads_daily, df_data=df_data, k=cutoff)
        if multiprocessing:
            pool = mp.Pool()
            ls_count = pool.map(partial_info, sta)
            pool.close()
            pool.join()
        else:
            ls_count = map(partial_info, sta)
        df_output = pd.concat(ls_count, axis=1, keys=[s.name for s in ls_count])
        df_output.to_csv(path_out, merge_cells=False)


if __name__ == "__main__":
    main()
