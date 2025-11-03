import numpy as np
from algs.state_window_ae import EpisodeWindowDataset
from algs.conv_ae import Conv1dStateSeqAutoencoder
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path

n_eps = 1000
ep_len = 1000
skip = 1
n_steps = ep_len // skip


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--lat', type=int)
    parser.add_argument('--hid', type=int)
    parser.add_argument('--no_fail', action='store_true')
    parser.add_argument('--window', type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
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
    s = s[:, ::skip]
    print(s.shape)
    
    if args.no_fail:  # train only on success data
        ds = []
        for i, val in enumerate(fail):
            if val == 0.:
                ds.append(s[i])
        ds = np.stack(ds, axis=0)
    else:
        ds = s

    print(f'Training on {len(ds)} samples.')

    dataset = EpisodeWindowDataset(ds, args.window)
    dl = DataLoader(dataset, batch_size=128, shuffle=True)

    model = Conv1dStateSeqAutoencoder(
        state_dim=s_dim,
        latent_dim=args.lat,
        hidden_dim=args.hid,
        lr=args.lr,
        window_len=args.window,
    )
    
    logger = TensorBoardLogger(dir/'conv')
    trainer = L.Trainer(max_epochs=3, logger=logger)

    trainer.fit(model, dl)
