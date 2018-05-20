import requests
import re
import pandas as pd
from datetime import date
import hydrobasics as hb
import os
import shutil
# import utilities as util


def set_nino(df_oai):
    """
    This function eval if a month is Nino, Nina or Neutral according to ONI.
    :param df_oai:
    :return:
    """
    sr_oni = df_oai['oni'].dropna()
    last_data_date = sr_oni.index.max()
    df_oai.loc[sr_oni.index, 'ENSO'] = 'Neutro'

    nino_range = pd.date_range(start='1950-01-01', end=last_data_date, freq='MS')

    for nino_date in nino_range:
        sr_nino_b = sr_oni.loc[:nino_date][-5:]
        sr_nino_f = sr_oni.loc[nino_date:][:5]

        if sr_nino_b[sr_nino_b >= .5].count() == 5:
            df_oai.loc[sr_nino_b.index, 'ENSO'] = 'Nino'

        elif sr_nino_f[sr_nino_f >= .5].count() == 5:
            df_oai.loc[sr_nino_f.index, 'ENSO'] = 'Nino'

        elif sr_nino_b[sr_nino_b <= -.5].count() == 5:
            df_oai.loc[sr_nino_b.index, 'ENSO'] = 'Nina'

        elif sr_nino_f[sr_nino_f <= -.5].count() == 5:
            df_oai.loc[sr_nino_f.index, 'ENSO'] = 'Nina'

        # else:
        #     df_oai.loc[nino_date, 'ENSO'] = 'Neutro'

    return df_oai


def fn_requests_download(url, path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def fn_oai2sr(link):
    name_oai = link[-(link[::-1].find('/')):]
    pre = 'http://www.esrl.noaa.gov'
    pos = '.data'
    name_file = 'tmp/' + name_oai + '.txt'
    # urllib.urlretrieve(pre + link + pos, name_file)
    fn_requests_download(pre + link + pos, name_file)
    file_oai = open(name_file, 'r')
    line_years = file_oai.readline().lstrip().rstrip()
    start_year = int(line_years[:4])
    stop_year = int(line_years[-4:])
    lines2read = stop_year - start_year + 1
    null_value = eval(file_oai.readlines()[lines2read].strip())
    print name_oai, start_year, stop_year, null_value
    mg_oai = pd.read_csv(name_file, index_col=0, skiprows=1, delim_whitespace=True, nrows=lines2read,
                         na_values=null_value, header=None)
    mg_oai.index.name = 'year'
    return hb.fn_mg2sr(mg_oai)


def fn_download_oai(links, date_data=date.today()):
    df_oai = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime(1948, 1, 1), end=date_data, freq='MS', name='Date'))

    for link in links:
        name_oai = link[-(link[::-1].find('/')):]
        try:
            df_oai[name_oai] = fn_oai2sr(link=link)

        except IOError:
            continue

    df_oai.sort_index(axis=1, inplace=True)

    return df_oai


def get_oai(oai_update=None, save=False):
    """
    This function gets a list (oai_update) or full ocean atmospheric indexes available in the NOAA website.
    :param oai_update: list of ocean atmospheric indexes for updating.
    :param save: save the dataframe into a excel file.
    :return: dataframe with ocean atmospheric indexes up to date.
    """
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # response = urllib2.urlopen('http://www.esrl.noaa.gov/psd/data/climateindices/list/')
    # html = response.read()
    html = requests.get('http://www.esrl.noaa.gov/psd/data/climateindices/list/', verify=True).text
    regex = ur'\href=\"(.+?)\.data\+?'
    links_noaa = re.findall(regex, html)
    today = date.today()

    if oai_update is None:
        df_oai = fn_download_oai(links=links_noaa, date_data=today)

    else:
        pref = links_noaa[0][:-(links_noaa[0][::-1].find('/'))]
        links_pre = [pref + link for link in oai_update]
        df_oai = fn_download_oai(links=links_pre, date_data=today)

    if save:
        name = 'OAIData_{year}{month:02}{day:02}.xlsx'.format(year=today.year, month=today.month, day=today.day)
        df_oai.to_excel(name, 'OAI', merge_cells=False)

    shutil.rmtree('tmp', ignore_errors=True)

    return df_oai


def estimate_oni(sr_nino34, fillin_date):
    """
    ONI estimation function.
    :param sr_nino34: Nino 34 series.
    :param fillin_date: Date for filling in ONI.
    :return:
    """
    def average30(year):
        if (year % 5) == 0:
            last_year_average = ((year - 1) // 5) * 5

        else:
            last_year_average = (year // 5) * 5

        first_year_average = last_year_average - 34
        return mg_nino_34.loc[first_year_average:last_year_average].mean()

    # sr_nino34 = get_oai(['nina34'])['nina34']
    # util.save_obj(sr_nino34, 'nino34', 'objs')
    # fill_date = pd.datetime(2016, 12, 01)
    # sr_nino34 = util.load_obj('nino34', 'objs')
    mg_nino_34 = hb.fn_sr2mg(sr_nino34)
    sr_average30 = average30(fillin_date.year)
    sr_sst34_sea = sr_nino34.rolling(window=3, min_periods=2, center=True).mean()
    return sr_sst34_sea.loc[fillin_date] - sr_average30.loc[fillin_date.month]


if __name__ == '__main__':
    get_oai()
