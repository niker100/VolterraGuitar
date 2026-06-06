import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models


# Load simulation data
def load_simulation_data(file_path):
    """
    Load time, input, and output voltage data from a text file.
    """
    with open(file_path) as file:
        data = file.read().strip().split("\n\n")

    time = []
    input_signal = []
    output_signal = []

    for block in data:
        lines = block.strip().split("\n")
        t = []
        input_sig = []
        output_sig = []
        for line in lines:
            values = line.split()
            t.append(float(values[0]))
            input_sig.append(float(values[1]))
            output_sig.append(float(values[2]))
        time.append(np.array(t))
        input_signal.append(np.array(input_sig))
        output_signal.append(np.array(output_sig))

    return time, input_signal, output_signal


# Prepare training data for the neural network
def prepare_training_data(input_signal, output_signal, memory_length=20):
    """
    Create training data for time-series prediction.
    """
    X, Y = [], []
    for input_sig, output_sig in zip(input_signal, output_signal):
        for i in range(memory_length, len(input_sig)):
            X.append(input_sig[i - memory_length : i])
            Y.append(output_sig[i])

    return np.array(X), np.array(Y)


# Build a neural network
def build_neural_network(memory_length, num_hidden_units=10000):
    """
    Build a feedforward neural network.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(memory_length,)),
            layers.Dense(num_hidden_units, activation="tanh"),
            layers.Dense(1, activation="linear"),  # Linear output layer
        ]
    )
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


# Extract Volterra kernels from the trained network
def extract_volterra_kernels(model, memory_length):
    """
    Extract Volterra kernels from the neural network weights.
    """
    weights = model.get_weights()
    input_to_hidden = weights[0]  # Weights from input to hidden layer
    hidden_biases = weights[1]  # Biases of hidden layer
    hidden_to_output = weights[2]  # Weights from hidden to output layer

    # First-order kernel
    h1 = np.sum(hidden_to_output * input_to_hidden.T, axis=1)

    # Second-order kernel
    h2 = np.zeros((memory_length, memory_length))
    for i in range(memory_length):
        for j in range(memory_length):
            h2[i, j] = np.sum(hidden_to_output * input_to_hidden[i, :] * input_to_hidden[j, :])

    # Third-order kernel
    h3 = np.zeros((memory_length, memory_length, memory_length))
    for i in range(memory_length):
        for j in range(memory_length):
            for k in range(memory_length):
                h3[i, j, k] = np.sum(
                    hidden_to_output
                    * input_to_hidden[i, :]
                    * input_to_hidden[j, :]
                    * input_to_hidden[k, :]
                )

    return h1, h2, h3


# Main script
if __name__ == "__main__":
    # Load data
    file_path = "circuit1.TNO"
    time, input_signal, output_signal = load_simulation_data(file_path)

    # Prepare training data
    memory_length = 100  # Number of past samples considered
    X, Y = prepare_training_data(input_signal, output_signal, memory_length)

    index = 100
    signal_length = len(X[index])
    print("Input signal length:", signal_length)
    t = np.linspace(0, signal_length, signal_length)
    Yvis = np.zeros_like(X[index])
    for i in range(0, signal_length):
        Yvis[i] = Y[index * signal_length + i]
    # Plot input and output signals
    plt.figure(figsize=(10, 5))
    plt.plot(t, X[index], label="Input")
    plt.plot(t, Yvis, label="Output")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.show()

    # Train the neural network
    model = build_neural_network(memory_length)
    model.fit(X, Y, epochs=20, batch_size=64, verbose=1)

    # Test NN with 440Hz sine wave
    t = np.linspace(0, 0.01, 300)
    x = 0.2 * np.sin(2 * np.pi * 1000 * t)
    y = np.zeros_like(x)
    for i in range(memory_length, len(x)):
        y[i] = model.predict(x[i - memory_length : i].reshape(1, -1))

    plt.figure(figsize=(10, 5))
    plt.plot(t, x, label="Input (440 Hz sine wave)")
    plt.plot(t, y, label="Output")
    plt.legend()
    plt.show()

    # Extract Volterra kernels
    h1, h2, h3 = extract_volterra_kernels(model, memory_length)

    # Save kernels to a file
    with open("stored.pckl", "wb") as f:
        pickle.dump(h1, f)
        pickle.dump(h2, f)
        pickle.dump(h3, f)

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(h1, label="First-order kernel")
    plt.title("First-order Kernel")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.imshow(h2, cmap="hot", interpolation="nearest")
    plt.title("Second-order Kernel")
    plt.colorbar(label="Amplitude")

    plt.subplot(1, 3, 3)
    plt.imshow(h3[:, :, memory_length // 2], cmap="hot", interpolation="nearest")
    plt.title("Third-order Kernel Slice")
    plt.colorbar(label="Amplitude")

    plt.show()
