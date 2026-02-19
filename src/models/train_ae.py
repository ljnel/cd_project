from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.pair_ae import PairAE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--lat', type=int)
    parser.add_argument('--hid', type=int)
    parser.add_argument('--no_fail', action='store_true')
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()

    s_dim, a_dim = 0, 0
    if args.env == 'inv_pend':
        s_dim, a_dim = 4, 1
    if args.env == 'hopper':
        s_dim, a_dim = 11, 3
    if args.env == 'half_cheetah':
        s_dim, a_dim = 17, 6
    if args.env == 'ant':
        s_dim, a_dim = 105, 8
    if args.env == 'humanoid':
        s_dim, a_dim = 348, 17
    dir = Path(f'./{args.env}')

    npz = np.load(dir/'train.npz')
    s, fail = npz['states'].astype(np.float32), npz['fail']

    if args.no_fail:  # train only on success data
        ds = []
        for i, val in enumerate(fail):
            if val == 0.:
                ds.append(s[i])
        ds = np.concatenate(ds, axis=0)
    else:
        ds = s.reshape(-1, s_dim)

    print(f'Training on {len(ds)} samples.')

    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)

    pae = PairAE(state_dim=s_dim, 
                 action_dim=0, 
                 latent_dim=args.lat, 
                 hidden_dim=args.hid,
                 lr=args.lr,
                 no_fail=args.no_fail)
    

    logger = TensorBoardLogger(dir/'ae')
    trainer = L.Trainer(max_epochs=5, logger=logger)

    trainer.fit(pae, dl)
