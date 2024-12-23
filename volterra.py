import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import sounddevice as sd
import soundfile as sf
from time import sleep
import pickle

def generate_test_signal(duration=0.004, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return t, signal

# Define kernels
def first_order_kernel():
    return np.array([0.5, 0.3, 0.2])  # Example linear kernel

def second_order_kernel():
    return np.array([[0.1, 0.05], [0.05, 0.02]])  # Example quadratic kernel

def echo_kernel(length=44100, delay=0.1, fs=44100, decay=0.5):
    """Generates a first-order kernel for an echo."""
    kernel = np.zeros(length)
    delay_samples = int(delay * fs)
    kernel[0] = 1.0  # Direct sound
    kernel[delay_samples] = decay  # Echo with decay
    return kernel

def multi_echo_kernel(length=44100, delays=[0.5, 0.8], decays=[0.5, 0.3], fs=44100):
    """Generates a kernel for multiple echoes."""
    kernel = np.zeros(length)
    for delay, decay in zip(delays, decays):
        delay_samples = int(delay * fs)
        kernel[delay_samples] += decay
    return kernel

def echo_second_order_kernel(length=10, decay=0.5):
    """Simulates second-order non-linear memory effects for echoes."""
    t = np.linspace(0, 1, length)
    kernel = np.outer(np.exp(-t), np.exp(-t)) * decay
    return kernel

def echo_third_order_kernel(length=5, decay=0.5):
    """Simulates third-order non-linear memory effects for echoes."""
    t = np.linspace(0, 1, length)
    kernel = np.outer(np.exp(-t), np.outer(np.exp(-t), np.exp(-t))) * decay
    return kernel

def compressor_first_order_kernel(length=100):
    """Generates a kernel for smoothing in compression."""
    t = np.linspace(0, 1, length)
    kernel = np.exp(-5 * t)  # Fast decay
    return kernel / np.sum(kernel)

def compressor_second_order_kernel(length=5, threshold=0.7):
    """Simulates a non-linear response for compression."""
    t = np.linspace(0, 1, length)
    kernel = np.outer(np.exp(-t), np.exp(-t))
    kernel[kernel > threshold] *= 0.5  # Apply non-linear attenuation
    return kernel

def bjt_first_order_kernel(length=100, f_low=50, f_high=5000, fs=44100):
    """
    First-order kernel approximating the linear bandpass response of a BJT amplifier.
    """
    t = np.linspace(0, length / fs, length)
    kernel = (np.exp(-2 * np.pi * f_low * t) - np.exp(-2 * np.pi * f_high * t))
    return kernel / np.sum(np.abs(kernel))  # Normalize

def bjt_second_order_kernel(length=50, saturation_level=1.0):
    """
    Second-order kernel for BJT saturation with quadratic non-linearity and memory effects.
    """
    t = np.linspace(0, 1, length)
    # Exponential decay for memory effects
    memory_kernel = np.exp(-2 * t)
    # Saturation effect (quadratic)
    kernel = np.outer(memory_kernel, memory_kernel) * saturation_level
    return kernel

def bjt_third_order_kernel(length=5, cubic_coeff=0.5):
    """
    Third-order kernel for cubic non-linearities in a BJT amplifier.
    """
    t = np.linspace(0, 1, length)
    # Memory effects with cubic interactions
    memory_kernel = np.exp(-3 * t)
    kernel = np.outer(memory_kernel, memory_kernel**2) * cubic_coeff
    return kernel

def volterra_series(input_signal, h1, h2=None, h3=None, alpha=0.1, beta=0.05):
    """
    Volterra Series implementation with first, second, and third-order terms.
    """
    # First-order response
    y1 = convolve(input_signal, h1, mode='full')[:len(input_signal)]
    
    # Second-order response
    y2 = np.zeros_like(input_signal)
    if h2 is not None:
        for tau1 in range(h2.shape[0]):
            for tau2 in range(h2.shape[1]):
                shift = tau1 + tau2
                if shift < len(input_signal):
                    y2[shift:] += h2[tau1, tau2] * input_signal[:len(input_signal) - shift] * input_signal[:len(input_signal) - shift]
        y2 /= np.max(np.abs(y2))  # Normalize
    
    # Third-order response
    y3 = np.zeros_like(input_signal)
    if h3 is not None:
        for tau1 in range(h3.shape[0]):
            for tau2 in range(h3.shape[1]):
                for tau3 in range(h3.shape[2]):
                    shift = tau1 + tau2 + tau3
                    if shift < len(input_signal):
                        y3[shift:] += h3[tau1, tau2, tau3] * input_signal[:len(input_signal) - shift]**3
        y3 /= np.max(np.abs(y3))  # Normalize
    
    # Combine terms with scaling
    y = y1 + alpha * y2 + beta * y3
    return y / np.max(np.abs(y))  # Normalize output


# Test the Volterra Series
# Generate a synthetic signal (e.g., sine wave)
t, x = generate_test_signal()

x = sf.read('full-reflective-guitar-chords-staccato-loop_115bpm_E_minor.wav')[0]
x = x[:, 0]
x = x / np.max(np.abs(x))
x = 10*x

# Define kernels
# h1 = echo_kernel()
# h2 = echo_second_order_kernel()
# h3 = echo_third_order_kernel()

f = open('stored.pckl', 'rb')
h1 = pickle.load(f)
h2 = pickle.load(f)
h3 = pickle.load(f)

# Apply Volterra Series
y = volterra_series(x, h1, h2, h3, alpha=1, beta=1)

# Plot and play the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x, label="Input Signal")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y, label="Output Signal")
plt.legend()
plt.show()

sd.play(y, samplerate=44100)
sd.wait()
