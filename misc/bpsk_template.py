# Import all necessary libraries.
import numpy as np
from scipy.special import erfc
from sklearn.naive_bayes import GaussianNB

# Number of BPSK symbols to be transmitted.
N = 1000000

# Instantiate a Gaussian naive Bayes classifier.
gnb = GaussianNB()

# Create Es/N0 vector.
EsN0dB = np.arange(-10,12,2)

ber_theo = ber_simu = np.zeros(len(EsN0dB))
for idx in range(0,len(EsN0dB)):
    EsN0Lin = 10.0**(-(EsN0dB[idx]/10.0))
    # Generate N BPSK symbols.
    x = (2.0 * (np.random.rand(N) >= 0.5) - 1.0).reshape(N, 1)
    # Generate noise vector. Divide by two once the theoretical ber uses a complex Normal pdf with variance of each part = 1/2.
    noise = np.sqrt(EsN0Lin/2.0)*np.random.randn(N, 1)
    # Pass symbols through AWGN channel.
    y = x + noise
    # Fit.
    gnb.fit(y, x.ravel())
    # Predict.
    detected_x = gnb.predict(y).reshape(N, 1)
    # Simulated BPSK BER.
    ber_simu[idx] = 1.0 * ((x != detected_x).sum()) / N
    # Theoretical BPSK BER.
    ber_theo[idx] = 0.5*erfc(np.sqrt(10.0**((EsN0dB[idx]/10.0))))