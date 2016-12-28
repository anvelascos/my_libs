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


if __name__ == '__main__':
    pass
