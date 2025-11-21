from benchmark.benchmark import *
from utils.paths import get_root

def test_process_data():
    ds = process_data(get_root() / 'benchmark' / 'HalfCheetah-v5' / 'train.npz')
    assert ds['s'].shape[0] == ds['a'].shape[0]