import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

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

# Define Volterra-inspired neural network
def volterra_layer(inputs, order=2):
    """Creates a Volterra-inspired layer for non-linear interactions."""
    linear_output = Dense(1, activation=None, name="linear")(inputs)
    
    # Create polynomial terms for Volterra kernel representation
    if order > 1:
        nonlinear_terms = []
        for i in range(order):
            power = i + 2
            term = Lambda(lambda x: tf.math.pow(x, power), name=f"power_{power}")(inputs)
            nonlinear_terms.append(Dense(1, activation=None, name=f"nonlinear_{power}")(term))
        nonlinear_output = Concatenate(name="nonlinear_output")(nonlinear_terms)
        return Concatenate(name="volterra_output")([linear_output, nonlinear_output])
    else:
        return linear_output


if __name__ == "__main__":
    time, X_train, y_train = load_simulation_data("circuit1.TNO")

    print(f"Number of samples: {len(X_train)}")
    print(f"Number of features: {len(X_train[100])}")
    # Model architecture
    input_dim = len(X_train[0])
    inputs = Input(shape=(input_dim,), name="input_layer")
    volterra_output = volterra_layer(inputs, order=1)  # Specify the Volterra kernel order
    output = Dense(1, activation=None, name="output_layer")(volterra_output)

    model = Model(inputs=inputs, outputs=output, name="Volterra_NN")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # Model summary
    #model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # Use 20% of the data for validation
        epochs=1,             # Number of epochs
        batch_size=4096,         # Batch size
        verbose=1              # Show training progress
    )

    # Save the model
    model.save("volterra_nn_model.h5")

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_train, y_train)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Extract Volterra kernels (trained weights)
    volterra_weights = {layer.name: layer.get_weights() for layer in model.layers if "linear" in layer.name or "nonlinear" in layer.name}
    print("Trained Volterra Kernels:")
    for name, weights in volterra_weights.items():
        print(f"{name}: {weights}")
