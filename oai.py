import urllib
import urllib2
import re
import pandas as pd
from datetime import date
import my_libs.hydrobasics as hb
import os
import shutil


def fn_oai2sr(link):
    name_oai = link[-(link[::-1].find('/')):]
    pre = 'http://www.esrl.noaa.gov'
    pos = '.data'
    name_file = 'tmp/' + name_oai + '.txt'
    urllib.urlretrieve(pre + link + pos, name_file)
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
        df_oai[name_oai] = fn_oai2sr(link=link)
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

    response = urllib2.urlopen('http://www.esrl.noaa.gov/psd/data/climateindices/list/')
    html = response.read()
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

    shutil.rmtree('tmp')

    return df_oai


if __name__ == '__main__':
    pass
