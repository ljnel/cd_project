def chebyshev_coefficients(fx):
    "Given samples of f at the Chebyshev points, compute"
    _, N = fx.shape
    coeffs = dct(fx, type=1) / (N - 1)
    coeffs[:, 0] /= 2
    coeffs[:, -1] /= 2
    assert 
    return coeffs

class ChebCD():
    """
    Fit on traj: (batch, chan, length)
    """

    def __init__(self):
        pass
        
    def fit(self):
        pass

    def predict(self):
        pass

if __name__ == '__main__':
    print('hi')


