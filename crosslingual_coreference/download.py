import hashlib
import inspect
import os
import urllib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from tqdm import tqdm

from .utils import ProgressBar, extract_single_file_from_zip

# shamelessly copied from https://github.com/alexandrainst/danlp/blob/master/danlp/download.py

DEFAULT_CACHE_DIR = os.getenv("DANLP_CACHE_DIR", os.path.join(str(Path.home()), ".danlp"))

DANLP_STORAGE_URL = 'http://danlp-downloads.alexandra.dk'

# The naming convention of the word embedding are on the form <dataset>.<lang>.<type>
# The <type> can be subword vectors=swv or word vectors=wv
MODELS = {
    # XLMR models
    'base': {
        'url': 'https://raw.githubusercontent.com/pandora-intelligence/crosslingual-coreference/models/english_roberta/model.tar.gz',
        'md5_checksum': '7cb9032c6b3a6af9d22f372de5817b35',
        'size': 853720929,
        'file_extension': '.tar.gz'
    }
}
class TqdmUpTo(tqdm):
    """
    This class provides callbacks/hooks in order to use tqdm with urllib.
    Read more here:
    https://github.com/tqdm/tqdm#hooks-and-callbacks
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize




def download_model(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR, process_func: Callable = None,
                   verbose: bool = False, clean_up_raw_data=True, force_download: bool = False, file_extension=None):
    """
    :param str model_name:
    :param str cache_dir: the directory for storing cached data
    :param process_func:
    :param bool verbose:
    :param bool clean_up_raw_data:
    :param bool force_download:
    :param str file_extension:
    """
    if model_name not in MODELS:
        raise ValueError("The model {} do not exist".format(model_name))

    model_info = MODELS[model_name]
    model_info['name'] = model_name

    model_file = model_name + model_info['file_extension'] if not file_extension else model_name + file_extension
    model_file_path = os.path.join(cache_dir, model_file)

    if not os.path.exists(model_file_path) or force_download:
        os.makedirs(cache_dir, exist_ok=True)

        _download_and_process(model_info, process_func, model_file_path, verbose)

    else:
        if verbose:
            print("Model {} exists in {}".format(model_name, model_file_path))

    return model_file_path


def _check_file(fname):
    """
    Method borrowed from
    https://github.com/fastai/fastai/blob/master/fastai/datasets.py
    :param fname:
    :return:
    """
    size = os.path.getsize(fname)
    with open(fname, "rb") as f:
        hash_nb = hashlib.md5(f.read(2 ** 20)).hexdigest()
    return size, hash_nb


def _check_process_func(process_func: Callable):
    """
    Checks that a process function takes the correct arguments
    :param process_func:
    """
    function_args = inspect.getfullargspec(process_func).args
    expected_args = ['tmp_file_path', 'meta_info', 'cache_dir', 'clean_up_raw_data', 'verbose']

    assert function_args[:len(expected_args)] == expected_args, "{} does not have the correct arguments".format(process_func)


def _download_and_process(meta_info: dict, process_func: Callable, single_file_path, verbose):
    """
    :param meta_info:
    :param process_func:
    :param single_file_path:
    :param verbose:
    """

    if process_func is not None:

        _check_process_func(process_func)

        tmp_file_path = NamedTemporaryFile().name
        _download_file(meta_info, tmp_file_path, verbose=verbose)

        cache_dir = os.path.split(single_file_path)[0]
        process_func(tmp_file_path, meta_info, cache_dir=cache_dir, verbose=verbose, clean_up_raw_data=True)

    else:
        single_file = meta_info['name'] + meta_info['file_extension']
        _download_file(meta_info, os.path.join(single_file_path, single_file), verbose=verbose)


def _download_file(meta_info: dict, destination: str, verbose: bool = False):
    """
    :param meta_info:
    :param destination:
    :param verbose:
    """
    file_name = os.path.split(destination)[1]

    expected_size = meta_info['size']
    expected_hash = meta_info['md5_checksum']
    url = meta_info['url']

    if not os.path.isfile(destination):
        if verbose:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1) as t:
                t.set_description("Downloading file {}".format(destination))
                urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
        else:
            print("Downloading file {}".format(destination))
            urllib.request.urlretrieve(url, destination, ProgressBar())

    else:
        if verbose:
            print("The file {} exists here: {}".format(file_name, destination))
            
    assert _check_file(destination) == (expected_size, expected_hash), \
        "Downloaded file does not match the expected size or checksum! Remove the file: {} and try again.".format(destination)


def _unzip_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                        clean_up_raw_data: bool = True, verbose: bool = False, file_in_zip: str = None):
    """
    Simple process function for processing models
    that only needs to be unzipped after download.

    :param str tmp_file_path: The path to the downloaded raw file
    :param dict meta_info:
    :param str cache_dir:
    :param bool clean_up_raw_data:
    :param bool verbose:
    :param str file_in_zip: Name of the model file in the zip, if the zip contains more than one file

    """
    from zipfile import ZipFile
    
    model_name = meta_info['name']
    full_path = os.path.join(cache_dir, model_name) + meta_info['file_extension']

    if verbose:
        print("Unzipping {} ".format(model_name))

    with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir
        
        file_list = zip_file.namelist()

        if len(file_list) == 1:
            extract_single_file_from_zip(cache_dir, file_list[0], full_path, zip_file)

        elif file_in_zip:
            extract_single_file_from_zip(cache_dir, file_in_zip, full_path, zip_file)

        else:  # Extract all the files to the name of the model/dataset
            destination = os.path.join(cache_dir, meta_info['name'])
            zip_file.extractall(path=destination)
