from sklearn.metrics import confusion_matrix

from data.datasets import load_experiment
from detectors.kernel import KernDetector

x_tr, x_te, y_true, _ = load_experiment('hopper')

model = KernDetector(kernel_type='fft', gamma=0.5, lam=1e-3, max_windows=100)
#model = ConvAEDetector(window=70, stride=10, lr=3e-4)
model.fit(x_tr)
y_pred = model.predict(x_te)
scores = confusion_matrix(y_true, y_pred, normalize='true').ravel()

#import matplotlib.pyplot as plt
#plt.hist(model.score_samples(x_te)[~y_true], color='green', alpha=0.3)
#plt.hist(model.score_samples(x_te)[y_true], color='red', alpha=0.3)
#plt.show()

print(f'TN: {scores[0] * 100:.2f}')
print(f'FP: {scores[1] * 100:.2f}')
print(f'FN: {scores[2] * 100:.2f}')
print(f'TP: {scores[3] * 100:.2f}')