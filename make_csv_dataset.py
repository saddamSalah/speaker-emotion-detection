import pandas as pd
import os
emotion_folders = ['Anger', 'Fear', 'Sad', 'Happy', 'Disgust', 'Neutral', 'Surprise']
#gender = 'female'


def make_filenaes_csv(path, emotion_folders):
    emotion_files = []
    emotion_file = []
    class_label = []
    pathes = []
    for fold in emotion_folders:
        emotion_files = os.listdir(path +  '/' + str(fold))
        emotion_file = [i for i in emotion_files if i.endswith('.wav')]

        for _file in range(len(emotion_file)):
            pathes.append(path  + str(fold)+'/'+str(emotion_file[_file]))
            class_label.append(fold)
    data = {}
    data['Path'] = pathes
    data['Label'] = class_label
    df = pd.DataFrame.from_dict(data)
    df.to_csv('training_files.csv', columns=['Path', 'Label'], index=False)


make_filenaes_csv('/data/wavenet/Saddam/sent_analysis_st2/', emotion_folders)


