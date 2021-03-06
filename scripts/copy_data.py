import csv
import os
from shutil import copyfile
import random

from logmmse import logmmse_from_file, logmmse
from pydub import AudioSegment
from scipy.io import wavfile

languages = [('nl', 'netherlands'), ('en', 'us'), ('de', 'germany'), ('fr', 'france'),
             ('es', 'nortepeninsular'),
             ('it', '')]

data_folder = '../trainingdata/'
gender = 'male'
for (lang, accent) in languages:
    src = 'samples/' + lang + '/'
    dest = 'selection/'
    tmp = 'tmpselection/'

    with open(data_folder + src + 'validated.tsv') as tsv_file:
        csv_reader = csv.reader(tsv_file, 'excel-tab')

        row_nr = 0
        cnt = 0
        approved_ids = []
        train_ids = []
        langs = []
        genders = []
        d_votes = []
        u_votes = []

        for row in csv_reader:
            train_ids.append(row[1])
            u_votes.append(row[3])
            d_votes.append(row[4])
            genders.append(row[6])
            langs.append(row[7])

        print(u_votes)

        for i in range(1, len(train_ids)):
            if int(u_votes[i]) >= 2 and int(d_votes[i]) <= 0 and langs[i] == accent \
                    and genders[i] == gender and os.path.exists(data_folder + src + train_ids[i]):
                approved_ids.append(train_ids[i])

        print(str(len(approved_ids)) + "  " + str(approved_ids))
        selected_ids = random.sample(approved_ids, 1)
        print(selected_ids)

        cnt = 0
        for selected in selected_ids:
            audio = AudioSegment.from_mp3(data_folder + src + selected)
            audio.export(data_folder + tmp + "tmp.wav", format="wav")

            rate, signal = wavfile.read(data_folder + tmp + "tmp.wav")
            out = logmmse(signal, rate)

            reduced = AudioSegment(
                signal.tobytes(),
                frame_rate=rate,
                sample_width=signal.dtype.itemsize,
                channels=1
            )
            reduced.export(data_folder + tmp + "reduced.mp3", format="mp3")

            #
            # copyfile(data_folder + src + selected,
            #          data_folder + dest + lang + '_' + str(cnt) + '.mp3')
            # cnt += 1
