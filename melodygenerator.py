import json
from tensorflow import keras
import numpy as np
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH


class CustomLSTM(keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        if "time_major" in kwargs:
            kwargs.pop("time_major")
        super().__init__(*args, **kwargs)


class MelodyGenerator:
    def __init__(self, model_path="./model_weight/lstm_model.h5"):
        self.model_path = model_path
        keras.utils.get_custom_objects().update({"LSTM": CustomLSTM})
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # create a seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(
                seed, num_classes=len(self._mappings)
            )
            # (1, max_sequence_length, num of symbols in the vocabulary)

            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction -> [0.1, 0.2, 0.1, 0.6]
            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temperature)

            # update the seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check if we're at the end of the melody
            if output_symbol == "/":
                break

            # update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
