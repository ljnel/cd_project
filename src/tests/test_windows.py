from utils.windows import make_windows
from utils.paths import get_root

import numpy as np

def test_windows():

    npz = np.load(get_root() / 'data/humanoid/train.npz')
    x, fail = npz['states'], npz['fail']

    fail = np.where(fail == 0, np.full_like(fail, -1), fail)
    print(f'{fail[fail > -1].min()}')

    win1, _ = make_windows(x, fail, window=50, horizon=20, verbose=True)
    win2, _ = make_windows(x, fail, window=40, horizon=20, verbose=True)
    assert len(win1) < len(win2)