# download dataset from http://www.esac-data.org/ , then put it in the "dataset" folder

If you want to see the notes in the staff, please do the following steps
# download musicHub from https://musescore.org/zh-hant/download, then install musicScore4

## type this code in your terminal:
from music21 import *
env = environment.Environment()
    
environment.Environment()['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
environment.Environment()['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

print('musicxmlPath:', environment.Environment()['musicxmlPath'])
print('musescoreDirectPNGPath:', environment.Environment()['musescoreDirectPNGPath'])
------------------------------------------------