import json
from tensorflow import keras
import numpy as np
import music21 as m21
from preprocessing import (
    preprocessing,
    create_single_file_dataset,
    create_mapping,
    SEQUENCE_LENGTH,
    MAPPING_PATH,
    KERN_DATASET_PATH,
    SAVE_DIR,
    SINGLE_FILE_DATASET,
)
from LSTM import train


class CustomLSTM(keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        if "time_major" in kwargs:
            kwargs.pop("time_major")
        super().__init__(*args, **kwargs)


class MelodyGenerator:
    def __init__(self, model_path="./LSTM/model_weight/lstm_test_model.h5"):
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

    def save_melody(
        self, melody, step_duration=0.25, format="midi", file_name="./LSTM/mel.mid"
    ):
        # create a music21 stream
        stream = m21.stream.Stream()

        # parse all the symbols in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're not dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = (
                        step_duration * step_counter
                    )  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol), quarterLength=quarter_length_duration
                        )

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":

    # preprocessing
    preprocessing(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)

    # train
    train()

    # melody generation
    mg = MelodyGenerator()
    seed = "79 _ 76 77 79 _ 76 77 79 67 69 71 72 74 76 77"
    seed2 = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = mg.generate_melody(seed2, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    mg.save_melody(melody)
