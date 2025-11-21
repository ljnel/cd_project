import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.fft import fft, fftfreq
from scipy.signal import sawtooth
from argparse import ArgumentParser

from utils.signals import *
from utils.paths import get_root
from algs.kern_cd import KernCD
from algs.kernels import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, fbeta_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    