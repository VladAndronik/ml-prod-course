import pandas as pd
import os
import h5py
from pathlib import Path
import xarray as xr
import numpy as np
from functools import partial
import time


save_dir = Path('tmp')
save_dir.mkdir(exist_ok=True)
root = Path('data_debug')

data_size = (10000, 5)
data_npy = np.random.uniform(size=data_size)
data = pd.DataFrame(data_npy, columns=(str(i) for i in range(data_npy.shape[1])))


def save_for_load():
    # csv
    os.makedirs(root / 'csv', exist_ok=True)
    data.to_csv(root / 'csv/dataset.csv', index=False)

    # parquet
    os.makedirs(root / 'parquet', exist_ok=True)
    data.to_parquet(root / 'parquet/dataset.parquet')

    # h5py
    os.makedirs(root / 'h5py', exist_ok=True)
    h5_file = h5py.File(root / 'h5py/dataset.h5', 'w')
    h5_file.create_dataset('dataset', data=data_npy)
    h5_file.close()

    # npy
    os.makedirs(root / 'npy', exist_ok=True)
    np.save(str(root / 'npy/dataset.npy'), data_npy)

    # xarray
    os.makedirs(root / 'xr', exist_ok=True)
    data.to_xarray().to_netcdf(root / 'xr/dataset.nc', engine='h5netcdf')


def load_csv(file):
    return pd.read_csv(file)


def load_npy(file):
    return np.load(file)


def load_h5(file):
    h5_file = h5py.File(file, 'r')
    data_array_h5 = h5_file['dataset'][()]
    h5_file.close()
    return data_array_h5


def load_xr(file):
    dataset_xarray = xr.open_dataset(file, engine='h5netcdf')
    dataset_netcdf4 = dataset_xarray.to_pandas()
    dataset_xarray.close()
    return dataset_netcdf4


def load_pq(file):
    return pd.read_parquet(file)


def save_csv(data, file):
    return data.to_csv(file, index=False)


def save_npy(data, file):
    return np.save(file, data)


def save_h5(data_array, file):
    h5_file = h5py.File(file, 'w')
    h5_file.create_dataset('data_array', data=data_array)
    h5_file.close()


def save_xr(data_array, file):
    xr.DataArray(data_array).to_netcdf(file, engine='h5netcdf')


def save_pq(data, file):
    data.to_parquet(file)


def measure_time(file, load_f):
    time1 = time.time()
    _ = load_f(file)
    time2 = time.time()

    return time2 - time1


def measure_time_no_arg(f):
    time1 = time.time()
    _ = f()
    time2 = time.time()

    return time2 - time1


def get_size_file(file):
    return os.stat(file).st_size / (1024 * 1024)


files = {
    'csv': (root / 'csv' / 'dataset.csv', load_csv,
            partial(save_csv, data=data, file=save_dir / 'csv.csv')),
    'h5': (root / 'h5py' / 'dataset.h5', load_h5,
           partial(save_h5, data_array=data_npy, file=save_dir / 'h5.h5')),
    'npy': (root / 'npy' / 'dataset.npy',
            load_npy, partial(save_csv, data=data, file=save_dir / 'npy.npy')),
    'xr': (root / 'xr' / 'dataset.nc', load_xr,
           partial(save_xr, data_array=data_npy, file=save_dir / 'xr.nc')),
    'parquet': (root / 'parquet' / 'dataset.parquet', load_pq,
                partial(save_pq, data=data, file=save_dir / 'pq.pq')),
}


if __name__ == '__main__':
    save_for_load()
    print(f"File format \t Size \t Load time \t Save time")
    for key, (file, load_fu, save_f) in files.items():
        print(key, get_size_file(file), measure_time(file, load_fu), measure_time_no_arg(save_f))
