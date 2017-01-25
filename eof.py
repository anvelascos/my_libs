import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import my_libs.hydrobasics as hb

# plt.style.use('ggplot')
# fontP = FontProperties()
# fontP.set_size('small')
# plt.style.context('seaborn-whitegrid')


def eof(df_input):
    # 4. Calculate the Covariance Matrix
    ix_data = df_input.columns
    ix_ampfun = range(1, ix_data.size + 1)
    df_cov = df_input.cov()
    print '\nCovariance Matrix:\n', df_cov.to_string()

    # 5. Solve Eigenvalues and Eigenvectors problem for covariances matrix
    na_evalues, na_evectors = np.linalg.eig(df_cov)

    ix_evalues = range(1, len(na_evalues) + 1)
    # sr_evalues = pd.Series(na_evalues, index=ix_evalues)
    df_evectors = pd.DataFrame(na_evectors, index=ix_evalues, columns=ix_evalues)
    df_results = pd.DataFrame(index=ix_ampfun, columns=['Eigenvalues', 'Cum_Variance', 'Rel_Variance'])
    df_results['Eigenvalues'] = pd.Series(na_evalues, index=ix_ampfun)
    df_results['Cum_Variance'] = df_results['Eigenvalues'].cumsum()
    df_results['Rel_Variance'] = df_results['Cum_Variance'] / df_results['Eigenvalues'].sum()
    print '\nExplained Variance:\n', df_results.to_string()
    print '\nEigenvectors:\n', df_evectors.to_string()
    # sub_covexp[n_dset].plot_adjust(df_results['Rel_Variance'])

    # 6. Calculate the Amplitude Functions (Principal Components)
    df_amplitude = pd.DataFrame(np.dot(df_input, na_evectors), index=df_input.index, columns=ix_ampfun)
    print '\nCovariance of Amplitude Functions:\n', df_amplitude.cov()
    print '\nAmplitude Functions:\n', df_amplitude.head(5).to_string()
    # wtp_af = [1, 2]
    # sub_ampfunc[n_dset].scatter(df_amplitude[wtp_af[0]], df_amplitude[wtp_af[1]])
    # sub_ampfunc[n_dset].set_xlabel('PC[' + str(wtp_af[0]) + ']')
    # sub_ampfunc[n_dset].set_ylabel('PC[' + str(wtp_af[1]) + ']')
    #
    # sub_ampfunc[n_dset].set_title('Principal Components for ' + data + ' data')
    #
    # wtp_ev = [1, 2]
    # texts = ix_data.values.astype(str)
    # x = df_evectors[wtp_ev[0]].values
    # y = df_evectors[wtp_ev[1]].values
    # sub_evector[n_dset].scatter(x, y)
    # sub_evector[n_dset].set_xlabel('PC[' + str(wtp_ev[0]) + ']')
    # sub_evector[n_dset].set_ylabel('PC[' + str(wtp_ev[1]) + ']')
    # sub_evector[n_dset].set_title('Principal Components for ' + data + ' data')
    #
    # for i, text in enumerate(texts):
    #     sub_evector[n_dset].annotate(text, (x[i], y[i]))
    #
    # df_amplitude.to_excel(xls_output, data, merge_cells=False)
    return df_amplitude


def fn_pca(df_feat, n_components=.85, std=True):
    if std:
        df_std = hb.fn_std(df_feat)
        df_std.dropna(how='all', axis=1, inplace=True)
    else:
        df_std = df_feat

    ix_train = df_feat.index
    pca = PCA(n_components=n_components)
    df_comp = pd.DataFrame(pca.fit_transform(df_std), index=ix_train, columns=range(1, pca.n_components_ + 1))

    return df_comp, pca.explained_variance_ratio_


def core_pca_corr(df_data, stations, folder_pca, par, zones, selstations=None, std=True, annotate=True, color=False,
                  sr_classgroup=None):
    # 1. PCA
    df_pca_tmp, exp_var = fn_pca(df_data.T, .999, std=std)
    df_pca, exp_var_tmp = fn_pca(df_data.T, 2, std=std)
    cumvar = np.cumsum(np.append([0], exp_var))
    sr_cumvar = pd.Series(cumvar, name='Varianza Explicada')
    ax_var = sr_cumvar.plot(title='Varianza Explicada PCA (Series {})'.format(zones))
    ax_var.set_ylabel('Varianza')
    ax_var.set_xlabel('Componente')
    figname = '{}/{}_PCA_Variance_{}'.format(folder_pca, par, zones)
    plt.savefig(figname)
    plt.close()

    if color:
        df_pca['c'] = sr_classgroup
        groups = df_pca.groupby('c')
        colors_names = colors.cnames.keys()
        colors2rem = [color2rem for color2rem in colors_names if ('white' in color2rem) or ('gray' in color2rem)]
        colors2rem = colors2rem + ['ivory', 'mintcream', 'azure', 'beige', 'seashell', 'lemonchiffon', 'oldlace',
                                   'aliceblue', 'linen', 'snow', 'cornsilk', 'lavenderblush', 'lightyellow',
                                   'honeydew']

        for x in colors2rem:
            colors_names.remove(x)

        colors_plot = iter(np.random.choice(colors_names, len(groups.groups)))
        fig, ax_pca = plt.subplots()

        for name, df_group in groups:
            color_plot = next(colors_plot)
            print('{}: {}'.format(name, color_plot))
            df_group.plot_adjust(kind='scatter', x=[1], y=[2], ax=ax_pca, color=color_plot, label=name)

        ax_pca.legend(prop=fontP)  # , loc='center left', bbox_to_anchor=(1, 0.5))

    else:
        ax_pca = df_pca.plot(kind='scatter', x=[1], y=[2], color='r')

    ax_pca.set_title('Analisis de Componentes Principales (Series {})'.format(zones))
    ax_pca.set_ylabel('Componente 2')
    ax_pca.set_xlabel('Componente 1')

    if annotate:
        if selstations is None:
            selstations = stations

        annotate_stations = stations[stations.isin(selstations)]

        for sta in annotate_stations:
            plt.annotate(sta, xy=(df_pca.loc[sta, 1], df_pca.loc[sta, 2]), )

    plt.axhline()
    plt.axvline()
    figname = '{}/{}_PCA_{}'.format(folder_pca, par, zones)
    plt.savefig(figname)
    plt.close()

    # 2. Correlation Analysis
    # 2.1 Correlation relationships
    df_corr = df_data.corr()
    rteo = hb.fn_rteo(df_data.shape[0] / 12)
    df_corr[df_corr.abs() < rteo] = None
    df_corr = pd.DataFrame(df_corr.unstack(), dtype=float).round(3)
    df_corr.dropna(inplace=True)
    df_corr.drop_duplicates(inplace=True)
    df_corr.drop(df_corr[df_corr[0] == 1].index, inplace=True)
    df_corr.index.names = ['Source', 'Target']
    df_corr.columns = ['Correlation']
    df_corr['ABS(Corr)'] = df_corr['Correlation'].abs()
    df_corr['Sign'] = df_corr['ABS(Corr)'] / df_corr['Correlation']
    # df_corr.to_excel(xls_output, '{}_Corr'.format(dict_areas[ah]), merge_cells=False)

    # 2.2 Nodes Correlation
    df_nodes = pd.DataFrame(index=stations)

    for sta_name in stations:
        sum_source = df_corr[df_corr.index.get_level_values('Source') == sta_name]['ABS(Corr)'].sum()
        sum_tarqet = df_corr[df_corr.index.get_level_values('Target') == sta_name]['ABS(Corr)'].sum()
        n_source = df_corr[df_corr.index.get_level_values('Source') == sta_name]['ABS(Corr)'].count()
        n_target = df_corr[df_corr.index.get_level_values('Target') == sta_name]['ABS(Corr)'].count()
        df_nodes.loc[sta_name, 'Cum_Corr'] = sum_source + sum_tarqet
        df_nodes.loc[sta_name, 'Connections'] = n_source + n_target
        df_nodes.loc[sta_name, 'Corr/Conn'] = (sum_source + sum_tarqet) / (n_source + n_target)
        df_nodes = df_nodes.round(3)

    df_nodes['Label'] = df_nodes.index.astype(str) + '\n(' + df_nodes['Corr/Conn'].astype(str) + ')'
    # df_nodes.to_excel(xls_output, '{}_Nodes'.format(dict_areas[ah]), merge_cells=False)

    return df_corr, df_nodes


if __name__ == '__main__':
    pass
