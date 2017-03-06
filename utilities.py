import cPickle as pickle
import os
import string


def save_obj(obj, name, folder='obj'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open('{}/{}.pkl'.format(folder, name), 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name, folder='obj'):
    with open('{}/{}.pkl'.format(folder, name), 'rb') as f:
        return pickle.load(f)


def core_makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_folders(project='project', subfolders=None):
    core_makedir(project)
    for subfolder in subfolders:
        core_makedir('{}/{}'.format(project, subfolder))


def adj_name(text):
    trans = string.maketrans(' ', '_')
    return text.translate(trans, '.')


def cutoff_data(df_data_cut, cutoff_date, w=None):
    """
    Cut a series or a dataframe taking into account a date and optionally (cutoff_date) a window (w)
    :param df_data_cut: data to cut off.
    :param cutoff_date: cut off date.
    :param w: window cut off size.
    :return:
    """
    if w is None:
        return df_data_cut.loc[:cutoff_date]

    else:
        return df_data_cut.loc[:cutoff_date].iloc[-w:]


if __name__ == '__main__':
    pass
