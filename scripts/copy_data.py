import csv
import os
from shutil import copyfile
import random

languages = [('nl', 'netherlands'), ('en', 'us'), ('de', 'germany'), ('fr', 'france'), ('es', 'nortepeninsular'),
         ('it', '')]

data_folder = '../trainingdata/'
gender = 'male'
for (lang, accent) in languages:
    src = 'samples/' + lang + '/'
    dest = 'tmpselection/'

    with open(data_folder + src + 'test.tsv') as tsv_file:
        csv_reader = csv.reader(tsv_file, 'excel-tab')

        row_nr = 0
        cnt = 0
        approved_ids = []
        train_ids = []
        langs = []
        genders = []

        for row in csv_reader:
            train_ids.append(row[1])
            langs.append(row[7])
            genders.append(row[6])

        if lang == 'it':
            s = set(langs)
            for l in s:
                print(l + " " + str(langs.count(l)))

        for i in range(len(train_ids)):
            if langs[i] == accent and genders[i] == gender and os.path.exists(
                    data_folder + src + train_ids[i]):
                approved_ids.append(train_ids[i])

        print(str(len(approved_ids)) + "  " + str(approved_ids))
        selected_ids = random.sample(approved_ids, 5)
        print(selected_ids)

        cnt = 0
        for selected in selected_ids:
            copyfile(data_folder + src + selected, data_folder + dest + lang + '_' + str(cnt) + '.mp3')
            cnt += 1
