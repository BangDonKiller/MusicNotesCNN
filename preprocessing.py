# download musicHub from https://musescore.org/zh-hant/download then install musicScore

from music21 import *
env = environment.Environment()
    
environment.Environment()['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
environment.Environment()['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

print('musicxmlPath:', environment.Environment()['musicxmlPath'])
print('musescoreDirectPNGPath:', environment.Environment()['musescoreDirectPNGPath'])

# -------------------------------------------

import os 
import music21 as m21

KERN_DATASET_PATH = 'datasets/deutschl/test'
ACCEPTABLE_DURATIONS = [
    0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4
]
SAVE_DIR = 'Dataset'

def load_songs_in_kern(data_path):
    songs = []
    for path, subdirs, files in os.walk(data_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))    
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
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
        key = song.analyze('key')
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
    
    for event in song.flat.notesAndRests:
        
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
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
                
def preprocessing(data_path):
    # load the folk song
    print('Loading songs...')
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f'Loaded {len(songs)} songs.')
    
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
    
if __name__ == '__main__':
    songs = load_songs_in_kern(KERN_DATASET_PATH)
    song = songs[0]
    
    preprocessing(KERN_DATASET_PATH)
    
    
    transposed_song = transpose(song)
    transposed_song.show()