"""module for loading datasets. taken from numpyro/examples

original: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/examples/datasets.py

slightly revised to use jax.random to provide randomness wherever needed
"""

import csv
import gzip
import os
import struct
from collections import namedtuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

from jax import device_put, lax
from jax.interpreters.xla import DeviceArray
import jax.numpy as jnp
import jax.random

from dppp.minibatch import split_batchify_data

if 'CI' in os.environ:
    DATA_DIR = os.path.expanduser('~/.data')
else:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '.data'))
os.makedirs(DATA_DIR, exist_ok=True)


dset = namedtuple('dset', ['name', 'urls'])


BASEBALL = dset('baseball', [
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/EfronMorrisBB.txt',
])


COVTYPE = dset('covtype', [
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/covtype.zip',
])


MNIST = dset('mnist', [
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/train-images-idx3-ubyte.gz',
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/train-labels-idx1-ubyte.gz',
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/t10k-images-idx3-ubyte.gz',
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/mnist/t10k-labels-idx1-ubyte.gz',
])


SP500 = dset('SP500', [
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/SP500.csv',
])


UCBADMIT = dset('ucbadmit', [
    'https://d2hg8soec8ck9v.cloudfront.net/datasets/UCBadmit.csv',
])


def _download(dset):
    for url in dset.urls:
        file = os.path.basename(urlparse(url).path)
        out_path = os.path.join(DATA_DIR, file)
        if not os.path.exists(out_path):
            print('Downloading - {}.'.format(url))
            urlretrieve(url, out_path)
            print('Download complete.')


def _load_baseball():
    _download(BASEBALL)

    def train_test_split(file):
        train, test, player_names = [], [], []
        with open(file, 'r') as f:
            csv_reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                player_names.append(row['FirstName'] + ' ' + row['LastName'])
                at_bats, hits = row['At-Bats'], row['Hits']
                train.append(np.array([int(at_bats), int(hits)]))
                season_at_bats, season_hits = row['SeasonAt-Bats'], row['SeasonHits']
                test.append(np.array([int(season_at_bats), int(season_hits)]))
        return np.stack(train), np.stack(test), np.array(player_names)

    train, test, player_names = train_test_split(os.path.join(DATA_DIR, 'EfronMorrisBB.txt'))
    return {'train': (train, player_names),
            'test': (test, player_names)}


def _load_covtype():
    _download(COVTYPE)

    file_path = os.path.join(DATA_DIR, 'covtype.data.gz')
    data = np.genfromtxt(gzip.GzipFile(file_path), delimiter=',')

    return {
        'train': (data[:, :-1], data[:, -1].astype(np.int32))
    }


def _load_mnist():
    _download(MNIST)

    def read_label(file):
        with gzip.open(file, 'rb') as f:
            f.read(8)
            data = np.frombuffer(f.read(), dtype=np.int8) / np.float32(255.)
            return device_put(data)

    def read_img(file):
        with gzip.open(file, 'rb') as f:
            _, _, nrows, ncols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8) / np.float32(255.)
            return device_put(data.reshape(-1, nrows, ncols))

    files = [os.path.join(DATA_DIR, os.path.basename(urlparse(url).path))
             for url in MNIST.urls]
    return {'train': (read_img(files[0]), read_label(files[1])),
            'test': (read_img(files[2]), read_label(files[3]))}


def _load_sp500():
    _download(SP500)

    date, value = [], []
    with open(os.path.join(DATA_DIR, 'SP500.csv'), 'r') as f:
        csv_reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            date.append(row['DATE'])
            value.append(float(row['VALUE']))
    date = np.stack(date)
    value = np.stack(value)

    return {'train': (date, value)}


def _load_ucbadmit():
    _download(UCBADMIT)

    dept, male, applications, admit = [], [], [], []
    with open(os.path.join(DATA_DIR, 'UCBadmit.csv')) as f:
        csv_reader = csv.DictReader(
            f,
            delimiter=';',
            fieldnames=['index', 'dept', 'gender', 'admit', 'reject', 'applications']
        )
        next(csv_reader)  # skip the first row
        for row in csv_reader:
            dept.append(ord(row['dept']) - ord('A'))
            male.append(row['gender'] == 'male')
            applications.append(int(row['applications']))
            admit.append(int(row['admit']))

    return {'train': (np.stack(dept), np.stack(male), np.stack(applications), np.stack(admit))}


def _load(dset):
    if dset == BASEBALL:
        return _load_baseball()
    elif dset == COVTYPE:
        return _load_covtype()
    elif dset == MNIST:
        return _load_mnist()
    elif dset == SP500:
        return _load_sp500()
    elif dset == UCBADMIT:
        return _load_ucbadmit()
    raise ValueError('Dataset - {} not found.'.format(dset.name))

def load_dataset(dset, batch_size=None, split='train', batchifier=split_batchify_data):
    """Loads a given dataset batchifies it with the given batchifier.
    """
    arrays = _load(dset)[split]
    size = len(arrays[0])
    if not batch_size:
        batch_size = size
    return batchifier(arrays, batch_size), size
