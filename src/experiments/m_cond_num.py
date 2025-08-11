from algs.cd_cheb import CDPolyCheb
from algs.cd_poly import CDPolynomial
from utils.plotting import save_plot

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

runs = 1

if __name__ == "__main__":
    df = []

    z = np.random.randn(10_000, 4)
    ds = range(1, 13)
    scaler = MinMaxScaler((-1, 1))
    z_scaled = scaler.fit_transform(z)

    for d in ds:
        for run in range(runs):
            # leave z as is
            p = CDPolynomial(z, degree=d)
            df.append({
                'degree': d,
                'scaling': False,
                'basis': 'mon',
                'cond': np.linalg.cond(p.moments),
                'run': run
            })

            pp = CDPolyCheb(z, degree=d)
            df.append({
                'degree': d,
                'scaling': False,
                'basis': 'cheb',
                'cond': np.linalg.cond(pp.M),
                'run': run
            })

            # scale to [-1, 1]^p
            p = CDPolynomial(z_scaled, degree=d)
            df.append({
                'degree': d,
                'scaling': True,
                'basis': 'mon',
                'cond': np.linalg.cond(p.moments),
                'run': run
            })

            pp = CDPolyCheb(z_scaled, degree=d)
            df.append({
                'degree': d,
                'scaling': True,
                'basis': 'cheb',
                'cond': np.linalg.cond(pp.M),
                'run': run
            })

    df = pd.DataFrame(df)
    sns.lineplot(df, x='degree', y='cond', hue='basis', style='scaling')
    plt.yscale('log')
    save_plot()