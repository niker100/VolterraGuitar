import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

# Load simulation data
def load_simulation_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')
    time, input_signal, output_signal = [], [], []
    for block in data:
        lines = block.strip().split('\n')
        t, input_sig, output_sig = [], [], []
        for line in lines:
            values = line.split()
            t.append(float(values[0]))
            input_sig.append(float(values[1]))
            output_sig.append(float(values[2]))
        time.append(np.array(t))
        input_signal.append(np.array(input_sig))
        output_signal.append(np.array(output_sig))
    return time, input_signal, output_signal

# Prepare training data
def prepare_training_data(input_signal, output_signal, memory_length):
    X, Y = [], []
    for input_sig, output_sig in zip(input_signal, output_signal):
        for i in range(memory_length, len(input_sig)):
            X.append(input_sig[i - memory_length:i])
            Y.append(output_sig[i])
    return np.array(X), np.array(Y)

# Hybrid CNN + RNN architecture
def build_hybrid_model(memory_length, hidden_units=128):
    inputs = layers.Input(shape=(memory_length,1))


    x = layers.Conv1D(128, 128, activation='tanh', padding='same')(inputs)
    
    # Fully connected output layer
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizers.Nadam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae'])
    return model


# Main script
if __name__ == "__main__":
    # Load data
    file_path = "circuit1.TNO"
    time, input_signal, output_signal = load_simulation_data(file_path)

    # Prepare data
    memory_length = 5
    X, Y = prepare_training_data(input_signal, output_signal, memory_length)

    # Build and train the hybrid model
    model = build_hybrid_model(memory_length, hidden_units=128)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        X, Y,
        epochs=100,
        batch_size=2048,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=True
    )

    # Test with a sine wave
    t = np.linspace(0, 0.01, 300)
    test_input = 0.2 * np.sin(2 * np.pi * 440 * t)
    test_output = np.zeros_like(test_input)
    for i in range(memory_length, len(test_input)):
        test_output[i] = model.predict(test_input[i - memory_length:i].reshape(1, memory_length, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(t, test_input, label="Input (440 Hz sine wave)")
    plt.plot(t, test_output, label="Output")
    plt.legend()
    plt.show()

    # Save model
    model.save("improved_model.h5")


    # # Extract and save Volterra kernels
    # h1, h2 = extract_volterra_kernels(model, memory_length)
    # with open("stored.pckl", "wb") as f:
    #     pickle.dump(h1, f)
    #     pickle.dump(h2, f)

    # # Plot Volterra kernels
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(h1, label="First-order Kernel")
    # plt.title("First-order Kernel")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.imshow(h2, cmap='hot', interpolation='nearest')
    # plt.title("Second-order Kernel")
    # plt.colorbar(label="Amplitude")
    # plt.show()

    # index_start = 300000
    # indices = [x for x in range(index_start, index_start+100)]
    # plt.figure(figsize=(15, 10))

    # for index in indices:
    #     signal_length = len(X[index])
    #     t = np.linspace(0, signal_length+len(indices), signal_length+len(indices))
    #     Xvis = np.zeros_like(t)
    #     Xvis[index-index_start:index-index_start+signal_length] = X[index]
    #     Yvis = np.zeros_like(t)
    #     Yvis[index-index_start:index-index_start+signal_length] = Y[index]
    #     # Plot each index on the same diagram
    #     plt.plot(t, Xvis, label=f"Input {index}")
    #     plt.scatter([index-index_start+signal_length], [Yvis[index-index_start]], label=f"Output {index-index_start}")

    # plt.xlabel("Time (s)")
    # plt.ylabel("Voltage (V)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

#     # Extract Volterra kernels
# def extract_volterra_kernels(model, memory_length):
#     weights = model.get_weights()
#     input_to_hidden = weights[0]  # input→hidden weights
#     hidden_biases = weights[1]    # hidden bias
#     hidden_to_output = weights[2] # hidden→output weights

#     # First-order kernel
#     h1 = np.sum(hidden_to_output * input_to_hidden.T, axis=1)

#     # Second-order kernel
#     h2 = np.zeros((memory_length, memory_length))
#     for i in range(memory_length):
#         for j in range(memory_length):
#             h2[i, j] = np.sum(hidden_to_output * input_to_hidden[i, :] * input_to_hidden[j, :])
#     return h1, h2