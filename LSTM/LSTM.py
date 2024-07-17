from tensorflow import keras
from preprocessing import (
    generate_training_sequences,
    SEQUENCE_LENGTH,
    dictionary_length,
)
import json


OUTPUT_UNITS = dictionary_length()
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "./LSTM/model_weight/lstm_test_model.h5"


def build_model(output_units, num_units, loss, lr):
    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)
    # compile the model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr), metrics=["accuracy"])

    # print the model information
    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, lr=LEARNING_RATE):
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    # build the network
    model = build_model(output_units, num_units, loss, lr)
    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # save the model
    model.save(SAVE_MODEL_PATH)
