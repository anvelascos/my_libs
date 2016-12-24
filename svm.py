import numpy as np
import spotpy
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

import my_libs.metrics as metrics
import my_libs.utilities as util


class SpotpySetup(object):
    def __init__(self, dict_params, x_train, y_train):

        self.params = [spotpy.parameter.Uniform(name=par_key, low=dict_params[par_key][0], high=dict_params[par_key][1])
                       for par_key in dict_params]
        self.x_train = x_train
        self.y_train = y_train

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, pars):
        x = np.array(pars)
        c = 2. ** x[0]
        # epsilon = 2. ** x[1]
        sigma = x[1]
        gamma = 1 / (2. * sigma ** 2)
        dict_pars = dict(C=c, gamma=gamma)
        return svr(x_train=self.x_train, y_train=self.y_train, pars=dict_pars)

    def evaluation(self):
        observations = self.y_train
        return observations

    def objectivefunction(self, simulation, observation):
        objectivefuncion = metrics.irmse0(obs=observation, sim=simulation)  # Negative for minimising
        return objectivefuncion


def svr(x_train, y_train, x_fore=None, pars=None):
    if x_fore is None:
        x_fore = x_train

    if pars is None:
        return SVR().fit(x_train, y_train).predict(x_fore)
    else:
        return SVR(C=pars['C'], gamma=pars['gamma']).fit(x_train, y_train).predict(x_fore)


def opt_spotpy(x_train, y_train, dict_params=None, rep=2000):
    x_train = x_train.dropna(axis=0)
    y_train = y_train.loc[x_train.index]
    print 'Optimising parameters with Spotpy...'
    if dict_params is None:
        dict_params = {'C': [-3., 3.], 'sigma': [0.05, 2.]}
    spotpy_setup = SpotpySetup(dict_params, x_train, y_train)
    sampler = spotpy.algorithms.sceua(spotpy_setup, dbname='ForeSCEUA', dbformat='csv', save_sim=False)
    sampler.sample(rep)
    results = sampler.getdata()
    best_pars = list(spotpy.analyser.get_best_parameterset(results)[0])
    dict_trans_pars = dict(C=2. ** best_pars[0], gamma=best_pars[1])
    return dict_trans_pars


def opt_sklearn_grid(x_train, y_train, grid_param=None):
    print '\n\nOptimising parameters with Scikit Learn...'
    if grid_param is None:
        r_c = np.logspace(-5, 5, 100, base=2)
        r_gamma = [1 / (2 * s ** 2) for s in np.arange(1e-2, 2., .04)]
        grid_param = dict(C=r_c, gamma=r_gamma)
    grid = GridSearchCV(SVR(), param_grid=grid_param)
    grid.fit(x_train, y_train)
    return grid.best_params_


def fn_svm(x_train, y_train, x_fore, pars=None, optpar=False, dict_pars=None, optmeth='spotpy'):
    """

    :param x_train:
    :param y_train:
    :param x_fore:
    :param pars: parameters for svm-rbf
    :param optpar: optimise parameters?
    :param dict_pars: dictionary for initialising the parameters optimisation
    :param optmeth: optimisation method ('spotpy' or 'sklearn').
    :return:
    """
    if optpar:
        optmeth = optmeth.lower()
        if optmeth == 'spotpy':
            best_pars = opt_spotpy(x_train=x_train, y_train=y_train, dict_params=dict_pars)
        elif optmeth == 'sklearn':
            best_pars = opt_sklearn_grid(x_train=x_train, y_train=y_train)
        else:
            best_pars = None
        return svr(x_train=x_train, y_train=y_train, x_fore=x_fore, pars=best_pars)
    else:
        return svr(x_train=x_train, y_train=y_train, x_fore=x_fore, pars=pars)


def main():
    data_train = util.load_obj('data_train_w060')
    x_train = data_train[0].values
    y_train = data_train[1].values

    best_par_spot = opt_spotpy(x_train, y_train)
    y_fore_best_spot = svr(x_train=x_train, y_train=y_train, pars=best_par_spot)
    mt_bestmodel_spot = metrics.Metrics(obs=y_train, sim=y_fore_best_spot)

    print '\nBest pars: {}\nr2: {}\nS/sd: {}\nRMSE: {}'.format(best_par_spot, mt_bestmodel_spot.r2,
                                                               mt_bestmodel_spot.ssd, mt_bestmodel_spot.rmse)

    best_par_skl = opt_sklearn_grid(x_train, y_train)
    y_fore_best_skl = svr(x_train=x_train, y_train=y_train, pars=best_par_skl)
    mt_bestmodel_skl = metrics.Metrics(obs=y_train, sim=y_fore_best_skl)
    print 'Best pars: {}\nr2: {}\nS/sd: {}\nRMSE: {}'.format(best_par_skl, mt_bestmodel_skl.r2,
                                                             mt_bestmodel_skl.ssd, mt_bestmodel_skl.rmse)


if __name__ == '__main__':
    main()
