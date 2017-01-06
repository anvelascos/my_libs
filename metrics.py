import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as pgs
plt.style.use('ggplot')

""" METRICAS ABSOLUTAS """


def ame(obs, sim):
    """
    Absolute Max Error
    :param obs: Observed data.
    :param sim: Simulated data.
    :return: ame value.
    """
    # diferencia = np.absolute(obs-sim)
    # resultado = diferencia.max()
    # return resultado
    return np.absolute(obs - sim).max()


def pdiff(obs, sim):
    """
    pdiff calcula la Diferencia de los valores maximos
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de pdiff
    """
    resultado = obs.max() - sim.max()
    return resultado


def mae(obs, sim):
    """
    mae calcula el error medio absoluto
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de mae
    """
    diferencia = np.absolute(obs - sim)
    resultado = np.mean(diferencia)

    return resultado


def merr(obs, sim):
    """
    merr calcula el error medio
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de merr
    """
    diferencia = (obs - sim)
    resultado = np.mean(diferencia)
    return resultado


def rmse(obs, sim):
    """
    rmse calcula la raiz cuadrada del error cuadratico medio
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de rmse
    """
    diferencia2 = (obs - sim) ** 2
    resultado = (np.sum(diferencia2) / (np.size(diferencia2))) ** 0.5
    return resultado


def r4ms4e(obs, sim):
    """
    r4ms4e calcula la raiz cuarta del error medio elevado a la 4 potencia
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de r4ms4e
    """
    diferencia2 = (obs - sim) ** 4
    resultado = (np.sum(diferencia2) / (np.size(diferencia2))) ** 0.25
    return resultado


def msle(obs, sim):
    """
    msle calcula el error cuadr?tico del error logar?tmico
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de msle
    """
    diferencia2 = (np.log(obs / sim)) ** 2
    resultado = (np.sum(diferencia2) / (np.size(diferencia2)))
    return resultado


def msde(obs, sim):
    """
    msde calcula el error medio cuadratico derivado
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de msde
    """
    diferencia = 0
    for i in range(1, len(obs)):
        diferencia = diferencia + ((obs[i] - obs[i - 1]) - (sim[i] - sim[i - 1])) ** 2
    resultado = diferencia / (np.size(obs) - 1)
    return resultado


""" METRICAS RELATIVAS """


def rae(obs, sim):
    """
    rae calcula el error absoluto relativo
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de rae
    """
    medobs = np.mean(obs)
    resultado = np.sum(abs(obs - sim)) / np.sum(abs(obs - medobs))
    return resultado


def pep(obs, sim):
    """
    pep calcula el error porcentual en pico
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de pep
    """
    resultado = 100 * (max(obs) - max(sim)) / max(obs)
    return resultado


def mare(obs, sim):
    """
    mare calcula el error medio absoluto relativo
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de mare
    """
    resultado = np.mean(np.absolute(obs - sim) / obs)
    return resultado


def mdape(obs, sim):
    """
    mdape calcula el error mediano absoluto en porcentaje
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de mdape
    """
    resultado = 100 * np.median(np.absolute(obs - sim) / obs)
    return resultado


def mre(obs, sim):
    """
    mre calcula el error relativo medio
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de mre
    """
    resultado = np.mean((obs - sim) / obs)
    return resultado


def msre(obs, sim):
    """
    msre calcula el error relativo medio cuadratico
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de msre
    """
    resultado = np.mean(((obs - sim) / obs) ** 2)
    return resultado


def rve(obs, sim):
    """
    rve calcula el error relativo volumetrico
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de rve
    """
    resultado = np.sum(obs - sim) / np.sum(obs)
    return resultado


""" METRICAS ADIMENSIONALES """


def rsqr(obs, sim):
    """
    rsqr calcula el coeficiente de determinacion
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de rsqr
    """
    medobs = np.mean(obs)
    medsim = np.mean(sim)
    numerador = np.sum((obs - medobs) * (sim - medsim))
    denominador = (np.sum((obs - medobs) ** 2) * np.sum((sim - medsim) ** 2)) ** 0.5
    resultado = (numerador / denominador) ** 2
    return resultado


def ioad(obs, sim):
    """
    ioad calcula el indice de aceptacion
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de ioad
    """
    medobs = np.mean(obs)
    numerador = np.sum((obs - sim) ** 2)
    denominador = np.sum((np.absolute(obs - medobs) + np.absolute(sim - medobs)) ** 2)
    resultado = 1 - (numerador / denominador)
    return resultado


def ce(obs, sim):
    """
    ce calcula el coeficiente de eficiencia
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de ce
    """
    medobs = np.mean(obs)
    numerador = np.sum((obs - sim) ** 2)
    denominador = np.sum((obs - medobs) ** 2)
    resultado = 1 - (numerador / denominador)
    return resultado


def pi(obs, sim):
    """
    pi calcula el indice de persistencia
    obs - es el vector de datos observados
    sim - es el vector de datos simulados
    retorna el valor de pi
    """
    numerador = np.sum((obs - sim) ** 2)
    denominador = 0
    for i in range(1, len(obs)):
        denominador = denominador + ((obs[i] - obs[i - 1])) ** 2
    resultado = 1 - numerador / denominador
    return resultado


def irmse0(obs, sim, t=1, obs_t=None):

    n=np.size(obs)
    delta=np.zeros(n - t)
    suma = 0
    for i in range(t, n):
        delta[i - t] = obs[i] - obs[i - t]
        suma = suma + delta[i - t]
    mdelta = suma / (n - t)
    suma = 0
    dif = np.zeros(np.size(delta))
    for i in range(np.size(delta)):
        dif[i]=(delta[i] - mdelta) ** 2
        suma = suma + dif[i]
    sigmad = (suma / (np.size(delta))) ** 0.5
    resultado = rmse(obs, sim) / sigmad
    return resultado


def ssd(obs, sim, t=1, obs_t=None):
    if obs_t is None:
        obs_t = obs.shift(-t)
        delta = obs - obs_t
        sigmad = np.std(delta)
    else:
        sigmad = np.std(obs - obs_t)

    resultado = rmse(obs, sim) / sigmad
    return resultado


def cshpr(obs, sim, t=1):
    n = np.size(obs)
    delta = obs[t:n] - obs[0:n - t]
    sigmad = np.std(delta)
    resultado = 0.674 * sigmad
    return resultado


def successes_p(obs, sim, bins=None, ret_shpr=True):
    """
    Successes percentage.
    :param obs: observed data
    :param sim: simulated data
    :param bins:
    :param ret_shpr: return value for Russian Hydrology Forecasting System
    :return:
    """
    if bins is None:
        bins = [.1, .2, .3, .4, .5]
    errors = np.abs((obs - sim) / obs)
    factor = np.digitize(errors, bins)
    successes = np.bincount(factor).cumsum() / float(len(errors))
    for i in range(len(successes), len(bins) + 1):
        successes = np.append(successes, 1.)
    if ret_shpr:
        shpr = np.sort(errors)[np.ceil(len(errors) * .7) - 2]
        return successes[:-1], shpr
    else:
        return successes[:-1]


def graph_met(obs_c, sim_c, sel_metrics, obs_v=None, sim_v=None, pt=None, title='Metrics', namefile='graph_metrics'):
    fmt = '{:.2f}'
    stex_suptitulo = 10
    stex_titulo = 9
    stex_subtitulo = 7
    stex_ejes = 6

    obs = pd.Series(index=(obs_c + obs_v).index)
    obs.loc[obs_c.index] = obs_c
    obs.loc[obs_v.index] = obs_v
    sim = pd.Series(index=obs.index)
    sim.loc[sim_c.index] = sim_c
    sim.loc[sim_v.index] = sim_v
    successp_c = successes_p(obs.loc[sim_c.index].values, sim_c.values)
    successp_v = successes_p(obs.loc[sim_v.index].values, sim_v.values)

    gs = pgs.GridSpec(26, 20, hspace=0, wspace=0)

    if not pt is None:
        # Precipitacion
        pt = pt.loc[obs.index]
        plt.subplot(gs[3:7, :])
        ax1 = pt.plot_adjust(kind='bar', xticks=[], secondary_y=True, color='blue', linewidth=0.1, label='PT')
        plt.ylabel('Precipitacion\n[mm/mes]', fontsize=stex_subtitulo)
        plt.xticks(fontsize=stex_ejes)
        plt.yticks(fontsize=stex_ejes)
        ax1.invert_yaxis()
        plt.grid(False)

    # Escorrentias observadas y simuladas
    plt.subplot(gs[7:14, :])
    obs.plot(style='-b', label='Y Observada')
    sim_c.plot_adjust(style='-r', label='Y Simulada Calibracion')
    sim_v.plot_adjust(style='--r', dashes=(3, 1), label='Y Simulada Validacion')
    plt.ylabel('Escorrentia\n[mm/mes]', fontsize=stex_subtitulo)
    plt.xlabel('')
    plt.xticks(fontsize=stex_ejes)
    plt.yticks(fontsize=stex_ejes)
    plt.grid(False)
    plt.legend(loc=2, fontsize=stex_ejes, ncol=3)

    # Diagrama de dispersion
    plt.subplot(gs[16:, 0:9])
    vmax = math.ceil(max(obs.max(), sim.max()))
    plt.scatter(obs.loc[sim_c.index], sim_c, marker='.', c='blue', label='Calibracion', s=10.0, edgecolors='none')
    plt.scatter(obs.loc[sim_v.index], sim_v, marker='.', c='red', label='Validacion', s=10.0, edgecolors='none')
    plt.plot([0, vmax], [0, vmax], '--k')
    plt.xlim(0, vmax)
    plt.ylim(0, vmax)
    plt.title('Diagrama de Dispersion', fontsize=stex_titulo)
    plt.ylabel('Escorrentia Simulada\n[mm/mes]', fontsize=stex_subtitulo)
    plt.xlabel('Escorrentia Observada\n[mm/mes]', fontsize=stex_subtitulo)
    plt.xticks(fontsize=stex_ejes)
    plt.yticks(fontsize=stex_ejes)
    plt.tick_params(width=.25, length=3.0)
    plt.grid(True, linewidth=.25)

    # Tabla de metricas de desempeno
    col_labels = ['Metrica', 'Calibracion', 'Validacion']
    table_vals = [[r'$S/\sigma_\Delta$', fmt.format(sel_metrics[0]), fmt.format(sel_metrics[3])],
                  [r'$R^2$', fmt.format(sel_metrics[1]), fmt.format(sel_metrics[4])],
                  [r'$RMSE$', fmt.format(sel_metrics[2]) + '%', fmt.format(sel_metrics[5]) + '%']]
    tabla = plt.table(cellText=table_vals, cellLoc='center', colWidths=[0.12, .15, .15],
                      colLabels=col_labels, loc='upper left')
    tabla.FONTSIZE = stex_ejes
    celdas = tabla.properties()['child_artists']
    tabla.figure.set_dpi(300)
    celdas[1]._text.set_color('blue')
    celdas[7]._text.set_color('red')
    for celda in celdas:
        celda.set_height(0.05)
        celda.set_linewidth(0.5)
        celda.set_alpha = 1.0
    tabla.set_zorder(100)

    # Porcentaje de aciertos
    plt.subplot(gs[16:, 11:])
    indice = np.arange(5)
    ancho = 0.2
    plt.bar(indice - ancho / 2, successp_c[0] * 100, align='center', width=ancho, color='blue')
    plt.bar(indice + ancho / 2, successp_v[0] * 100, align='center', width=ancho, color='red')
    plt.xticks(range(5), ['10%', '20%', '30%', '40%', '50%'], fontsize=stex_subtitulo)
    plt.title('Porcentaje de Aciertos', fontsize=stex_titulo)
    plt.xlabel('Error\n[%]', fontsize=stex_subtitulo)
    plt.ylabel('Aciertos\n[%]', fontsize=stex_subtitulo)
    plt.xticks(fontsize=stex_ejes)
    plt.yticks(range(0, 110, 10), fontsize=stex_ejes)
    plt.tick_params(width=.25, length=3.0)
    plt.grid(True, linewidth=.25)

    # Tabla criterio sistema hidrologico de pronostico ruso (shpr)
    col_labels = ['%Aciertos', 'Calibracion', 'Validacion']
    table_vals = [[r'70%', fmt.format(successp_c[1] * 100) + '%', fmt.format(successp_v[1] * 100) + '%']]
    tabla = plt.table(cellText=table_vals, cellLoc='center', colWidths=[0.12, .15, .15],
                      colLabels=col_labels, loc='upper left')
    tabla.FONTSIZE = stex_ejes
    celdas = tabla.properties()['child_artists']
    tabla.figure.set_dpi(300)
    celdas[3]._text.set_color('red')
    celdas[1]._text.set_color('blue')
    for celda in celdas:
        celda.set_height(0.05)
        celda.set_linewidth(0.5)
        celda.set_alpha = 1.0
    tabla.set_zorder(100)

    # titulos y guardar
    plt.suptitle(title, fontsize=stex_suptitulo)
    plt.tight_layout()
    plt.savefig('Resultados/Graficas/' + namefile + '.jpg', dpi=300, orientation='landscape', loc='center',
                papertype='letter')
    plt.close()


class Metrics(object):
    def __init__(self, obs, sim, t=1, obs_t=None):
        self.obs = obs
        self.sim = sim
        self.r2 = rsqr(obs=obs, sim=sim)

        try:
            self.ssd = ssd(obs=obs, sim=sim, t=t, obs_t=obs_t)
        except AttributeError:
            self.ssd = irmse0(obs=obs, sim=sim, t=t)

        # self.rmse = rmse(obs=obs, sim=sim)
        self.mare = mare(obs=obs, sim=sim)
        self.rmse = rmse(obs=obs, sim=sim)
        self.success_p, self.shpr = successes_p(obs=obs, sim=sim, ret_shpr=True)
        # self.me = merr(obs=obs, sim=sim)


if __name__ == "__main__":
    pass
