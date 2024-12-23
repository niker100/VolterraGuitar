import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import keras_tuner as kt

# Custom layer for orthogonal activations
class DenseOrthogonal(layers.Layer):
    def __init__(self, units, activation_functions, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_functions = activation_functions
        if len(activation_functions) != units:
            raise ValueError("Number of activation functions must match the number of units.")

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        linear_output = K.dot(inputs, self.kernel) + self.bias
        output = []
        for i, activation in enumerate(self.activation_functions):
            output.append(activation(linear_output[:, i]))
        return K.stack(output, axis=-1)

# Legendre Polynomial activation functions
def l1(x):
    return x

def l2(x):
    return 0.5 * (3 * x ** 2 - 1)

def l3(x):
    return 0.5 * (5 * x ** 3 - 3 * x)

def l4(x):
    return 0.125 * (35 * x ** 4 - 30 * x ** 2 + 3)

def l5(x):
    return 0.125 * (63 * x ** 5 - 70 * x ** 3 + 15 * x)

def l6(x):
    return 0.0625 * (231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5)

def l7(x):
    return 0.0625 * (429 * x ** 7 - 693 * x ** 5 + 315 * x ** 3 - 35 * x)

def l8(x):
    return 0.0078125 * (6435 * x ** 8 - 12012 * x ** 6 + 6930 * x ** 4 - 1260 * x ** 2 + 35)

def l9(x):
    return 0.0078125 * (12155 * x ** 9 - 25740 * x ** 7 + 18018 * x ** 5 - 4620 * x ** 3 + 315 * x)

def l10(x):
    return 0.00390625 * (46189 * x ** 10 - 109395 * x ** 8 + 90090 * x ** 6 - 30030 * x ** 4 + 3465 * x ** 2 - 63)

def l11(x):
    return 0.00390625 * (88179 * x ** 11 - 230945 * x ** 9 + 218790 * x ** 7 - 90090 * x ** 5 + 15015 * x ** 3 - 693 * x)

def l12(x):
    return 0.0009765625 * (676039 * x ** 12 - 1939938 * x ** 10 + 2078505 * x ** 8 - 1021020 * x ** 6 + 225225 * x ** 4 - 18018 * x ** 2 + 231)

def l13(x):
    return 0.0009765625 * (1300075 * x ** 13 - 4061376 * x ** 11 + 4686825 * x ** 9 - 2552550 * x ** 7 + 675675 * x ** 5 - 90090 * x ** 3 + 3003 * x)

def l14(x):
    return 0.000244140625 * (5014575 * x ** 14 - 16900975 * x ** 12 + 22309287 * x ** 10 - 14549535 * x ** 8 + 4849845 * x ** 6 - 765765 * x ** 4 + 45045 * x ** 2 - 429)

def l15(x):
    return 0.000244140625 * (9694845 * x ** 15 - 35102025 * x ** 13 + 50702925 * x ** 11 - 37182145 * x ** 9 + 14549535 * x ** 7 - 2909907 * x ** 5 + 255255 * x ** 3 - 6435 * x)

def l16(x):
    return 6.103515625e-05 * (300540195 * x ** 16 - 1163381400 * x ** 14 + 1825305300 * x ** 12 - 1487285800 * x ** 10 + 669278610 * x ** 8 - 167319652 * x ** 6 + 22309287 * x ** 4 - 1214850 * x ** 2 + 6435)

def l17(x):
    return 6.103515625e-05 * (583401555 * x ** 17 - 2404321560 * x ** 15 + 4094773230 * x ** 13 - 3412310590 * x ** 11 + 1571343270 * x ** 9 - 418965090 * x ** 7 + 58340155 * x ** 5 - 3876035 * x ** 3 + 6435 * x)

def l18(x):
    return 1.52587890625e-05 * (34459425 * x ** 18 - 155117520 * x ** 16 + 290990700 * x ** 14 - 267711444 * x ** 12 + 134217728 * x ** 10 - 37182145 * x ** 8 + 5495490 * x ** 6 - 383838 * x ** 4 + 10935 * x ** 2 - 715)

def l19(x):
    return 1.52587890625e-05 * (672822343 * x ** 19 - 3190187286 * x ** 17 + 6466264220 * x ** 15 - 6183775204 * x ** 13 + 3357800060 * x ** 11 - 1047765125 * x ** 9 + 180180180 * x ** 7 - 16016157 * x ** 5 + 658008 * x ** 3 - 10935 * x)

def l20(x):
    return 3.814697265625e-06 * (393822575 * x ** 20 - 1969112875 * x ** 18 + 4257795750 * x ** 16 - 4508102925 * x ** 14 + 2646692685 * x ** 12 - 891080190 * x ** 10 + 169009750 * x ** 8 - 18018018 * x ** 6 + 945945 * x ** 4 - 18018 * x ** 2 + 121)

def l21(x):
    return 3.814697265625e-06 * (775587605 * x ** 21 - 4088353405 * x ** 19 + 9256107870 * x ** 17 - 10477651250 * x ** 15 + 6374081615 * x ** 13 - 2186189404 * x ** 11 + 405134676 * x ** 9 - 39671357 * x ** 7 + 1939938 * x ** 5 - 45045 * x ** 3 + 231 * x)

def l22(x):
    return 9.5367431640625e-07 * (4750104245 * x ** 22 - 25518731270 * x ** 20 + 58921019140 * x ** 18 - 70288497250 * x ** 16 + 46532348200 * x ** 14 - 18032411700 * x ** 12 + 4051346760 * x ** 10 - 50491812 * x ** 8 + 3116130 * x ** 6 - 90090 * x ** 4 + 1155 * x ** 2 - 231)

def l23(x):
    return 9.5367431640625e-07 * (9315548320 * x ** 23 - 51173876080 * x ** 21 + 123134292140 * x ** 19 - 156764640200 * x ** 17 + 111974787600 * x ** 15 - 46532348200 * x ** 13 + 10804325340 * x ** 11 - 1350540668 * x ** 9 + 82598880 * x ** 7 - 2312310 * x ** 5 + 27027 * x ** 3 - 1155 * x)

def l24(x):
    return 2.384185791015625e-07 * (5810470175 * x ** 24 - 32447658500 * x ** 22 + 79960182900 * x ** 20 - 107408279200 * x ** 18 + 82306502400 * x ** 16 - 37182145000 * x ** 14 + 10039179150 * x ** 12 - 1551175200 * x ** 10 + 128432040 * x ** 8 - 5220075 * x ** 6 + 90090 * x ** 4 - 462 * x ** 2 + 253)

def l25(x):
    return 2.384185791015625e-07 * (11668031100 * x ** 25 - 65662208400 * x ** 23 + 164155521000 * x ** 21 - 225952003200 * x ** 19 + 176446179000 * x ** 17 - 82306502400 * x ** 15 + 22830046200 * x ** 13 - 3649536000 * x ** 11 + 319018728 * x ** 9 - 14158140 * x ** 7 + 270270 * x ** 5 - 2002 * x ** 3 + 253 * x)

def l26(x):
    return 5.960464477539063e-08 * (74974312500 * x ** 26 - 426093023200 * x ** 24 + 1094533445920 * x ** 22 - 1551175200000 * x ** 20 + 1277771184000 * x ** 18 - 635306592000 * x ** 16 + 194594425000 * x ** 14 - 36495360000 * x ** 12 + 4051346760 * x ** 10 - 248063644 * x ** 8 + 8259888 * x ** 6 - 135135 * x ** 4 + 858 * x ** 2 - 253)

def l27(x):
    return 5.960464477539063e-08 * (150845043300 * x ** 27 - 865432765200 * x ** 25 + 2268783828000 * x ** 23 - 3283101056000 * x ** 21 + 2807774372000 * x ** 19 - 1527832968000 * x ** 17 + 529284144200 * x ** 15 - 109453344592 * x ** 13 + 12777711840 * x ** 11 - 777177030 * x ** 9 + 24806364 * x ** 7 - 378378 * x ** 5 + 2574 * x ** 3 - 858 * x)

def l28(x):
    return 1.4901161193847656e-08 * (105204948186 * x ** 28 - 609749174800 * x ** 26 + 1629547924800 * x ** 24 - 2406725880000 * x ** 22 + 2104098968000 * x ** 20 - 1163381400000 * x ** 18 + 411567167400 * x ** 16 - 89508428480 * x ** 14 + 11732745000 * x ** 12 - 874125600 * x ** 10 + 34234200 * x ** 8 - 630630 * x ** 6 + 45045 * x ** 4 - 715 * x ** 2 + 253)

def l29(x):
    return 1.4901161193847656e-08 * (211915132832 * x ** 29 - 1236669875976 * x ** 27 + 3357800060000 * x ** 25 - 5049181200000 * x ** 23 + 4469737680000 * x ** 21 - 2510732976000 * x ** 19 + 895084284800 * x ** 17 - 195084841536 * x ** 15 + 25337893200 * x ** 13 - 1874251200 * x ** 11 + 74207304 * x ** 9 - 1261260 * x ** 7 + 90090 * x ** 5 - 715 * x ** 3 + 715 * x)

def l30(x):
    return 3.725290298461914e-09 * (135207850050 * x ** 30 - 792594609780 * x ** 28 + 2186189404000 * x ** 26 - 3346393056000 * x ** 24 + 3016757168000 * x ** 22 - 1673196520000 * x ** 20 + 595665404400 * x ** 18 - 130945815600 * x ** 16 + 16900975000 * x ** 14 - 1247721600 * x ** 12 + 49521900 * x ** 10 - 756756 * x ** 8 + 45045 * x ** 6 - 715 * x ** 4 + 715 * x ** 2 - 253)

# Activation functions
activation_functions = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29, l30]

# Load simulation data
def load_simulation_data(file_path):
    """
    Load time, input, and output voltage data from a text file.
    """
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')

    time = []
    input_signal = []
    output_signal = []

    for block in data:
        lines = block.strip().split('\n')
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

# Prepare training data
def prepare_training_data(input_signal, output_signal, memory_length):
    """
    Create training data for time-series prediction.
    """
    X, Y = [], []
    for input_sig, output_sig in zip(input_signal, output_signal):
        input_sig = np.concatenate([input_sig, input_sig[:memory_length]])
        output_sig = np.concatenate([output_sig, output_sig[:memory_length]])

        for i in range(memory_length, len(input_sig)):
            X.append(input_sig[i - memory_length:i])
            Y.append(output_sig[i])

    return np.array(X), np.array(Y)

# Build the model
def build_model(memory_length):
    model = models.Sequential([
        layers.Input(shape=(memory_length,)),
        DenseOrthogonal(units=len(activation_functions), activation_functions=activation_functions),
        layers.Dense(1, activation='linear'),  # Linear output layer
    ])
    model.compile(optimizer='nadam', loss='mse', metrics=['mae'])
    return model

# Main script
if __name__ == "__main__":
    # Load data
    file_path = "circuit1.TNO"
    time, input_signal, output_signal = load_simulation_data(file_path)

    # Prepare training data
    memory_length = 20
    X, Y = prepare_training_data(input_signal, output_signal, memory_length)

    # Build and train the model
    model = build_model(memory_length)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    model.fit(X, Y, epochs=1000, batch_size=4096, verbose=1) #, validation_split=0.2, callbacks=[early_stopping])

    # Test NN with a sine wave
    t = np.linspace(0, 0.01, 300)
    x = 0.2 * np.sin(2 * np.pi * 1000 * t)
    y = np.zeros_like(x)
    for i in range(memory_length, len(x)):
        y[i] = model.predict(x[i - memory_length:i].reshape(1, -1))

    plt.figure(figsize=(10, 5))
    plt.plot(t, x, label="Input (440 Hz sine wave)")
    plt.plot(t, y, label="Output")
    plt.legend()
    plt.show()
