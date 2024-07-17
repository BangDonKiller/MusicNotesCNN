import os
import music21 as m21
import json
import keras
import numpy as np

KERN_DATASET_PATH = "./LSTM/dataset/deutschl/test"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
SAVE_DIR = "./LSTM/encoded_dataset"
SINGLE_FILE_DATASET = "./LSTM/encoded_dataset/file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "./LSTM/encoded_dataset/mapping.json"


def load_songs_in_kern(data_path):
    songs = []
    for path, subdirs, files in os.walk(data_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flatten().notesAndRests:
        if note.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    print(key)

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)
    return tranposed_song


def encode_song(song, time_step=0.25):
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]

    encoded_song = []

    for event in song.flatten().notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocessing(dataset_path):
    # load the folk song
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        # filter out songs that have non-accepable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # transpose song to C major / A minor
        song = transpose(song)

        # encode songs with time series representation
        encoded_song = encode_song(song)

        # save song to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as file:
            file.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as file:
        song = file.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]

    # save string that contains all dataset
    if not os.path.exists(file_dataset_path):
        with open(file_dataset_path, "w+") as f:
            f.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to a json file
    if not os.path.exists(mapping_path):
        with open(mapping_path, "w") as f:
            json.dump(mappings, f, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as f:
        mappings = json.load(f)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    # if input length is 2
    # [11, 12, 13, 14, ...] -> input: [11, 12], target:13; input: [12, 13], target: 14

    # load songs and map them to int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    # 100 symbols, 64 sequence length -> 100 - 64 = 36 input
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i : i + sequence_length])
        targets.append(int_songs[i + sequence_length])

    # one-hot encode the sequences
    # inputs shape: (num_sequences, sequence_length) -> (num_sequences, sequence_length, vocabulary_size)
    # [11, 12, 13] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def dictionary_length():
    with open("./LSTM/encoded_dataset/mapping.json", "r") as fp:
        mappings = json.load(fp)
    return len(mappings)
