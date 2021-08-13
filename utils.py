from pathlib import Path
import os
import glob
import codecs
import json
import shutil
import logging
import torch
import random
import numpy as np
import _pickle as cPickle


def seed_all(seed_value, gpu):
    """
    Set the seed value for python, numpy and torch.
    """
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if gpu:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars


def get_filename(path):
    return os.path.basename(path).split('.')[0]


def dirname(path):
    return os.path.dirname(path)


def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)


def get_files(path, ext):
    return list(set(glob.glob(path + '*.' + ext) + glob.glob(path + '*.' + ext.upper())))


def get_nested_files(path, ext):
    files = []
    for f_name in Path(path).glob('**/*.{}'.format(ext)):
        files.append(f_name)
    return files


def filter_files(files, filter_str):
    filtered_files = []
    for file in files:
        if filter_str in file:
            filtered_files.append(file)
    return filtered_files


def resource_exists(f_path):
    if not f_path:
        return False
    if os.path.exists(f_path):
        return True
    return False


def delete_file(path):
    os.remove(path)


def read_pickle(path):
    with open(path, 'rb') as f:
        return cPickle.load(f)


def write_pickle(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
    print('{} created'.format(path))


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_directory(path):
    shutil.rmtree(path)


def get_parent_directory(path):
    return os.path.dirname(path)


def write_json(data, path):
    with codecs.open(path, 'w', 'utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True)
        print('{} created'.format(path))


def set_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # logging to a file
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def namespace_to_dict(args):
    params = dict()
    params["train"] = args.train_path
    params["dev"] = args.dev_path
    params["tag_scheme"] = args.tag_scheme
    params["lower"] = args.lower
    params["zeros"] = args.zeros
    params["char_dim"] = args.char_dim
    params["char_lstm_dim"] = args.char_lstm_dim
    params["char_bidirect"] = args.char_bidirect
    params["word_dim"] = args.word_dim
    params["word_lstm_dim"] = args.word_lstm_dim
    params["word_bidirect"] = args.word_bidirect
    params["embeddings_path"] = args.embeddings_path
    params["embeddings_pkl_path"] = args.embeddings_pkl_path
    params["all_emb"] = args.all_emb
    params["cap_dim"] = args.cap_dim
    params["crf"] = args.crf
    params["dropout"] = args.dropout
    params["lr_method"] = args.lr_method
    params["pos_dim"] = args.pos_dim
    params["ortho_dim"] = args.ortho_dim
    params["multi_task"] = args.multi_task
    params["ranking_loss"] = args.ranking_loss
    params["language_model"] = args.language_model
    params["seed"] = args.seed
    params["patience"] = args.patience

    return params
