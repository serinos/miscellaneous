"""
(Respiratory Sound Diagnosis)
Aug 18 2019 - Merged version of plotter and database dumper scripts
I could use part of the code again for some sound project maybe, so here it is...

Takes in wav files in the same folder and patient_diagnosis.csv
Outputs database as database_new.csv and spectrograms as [soundfile name].pdf
The outputs are resultant from short time fourier transform of the first 10 seconds of
the sound files, and only 60-2000Hz is taken
Each stft dump of sound files are preceded by info on the file:
[Index, Healthy, COPD, Pneumonia, URTI, Bronchiectasis, Bronchiolitis]
Index starts from 1, other variables are assigned 1 or 0 depending on diagnoses
If sound file is less than 10 seconds, it is skipped and skipped files are
noted in skipped_data.txt; also, index : filename is saved in index_codes.txt
when dumping the database
Database has 1 row of header followed by 60 rows of fourier transform data for 915 files
Header format: [Index(0:916), Healthy(1)/not(0), COPD(1)/not(0), Pneumonia(1)/not(0), URTI(1)/not(0), Bronchiectasis(1)/not(0), Bronchiolitis(1)/not(0)]

Use pdfunite from poppler-utils on *nix to merge pdfs: "pdfunite *.pdf merged.pdf"
"""

# 3x3 compression may be too much, experiment with 2x2 and also the original STFT outputs

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
fig = plt.figure()


def plotter(D, filename, diagnosis):
    librosa.display.specshow(D)
    plt.title(filename+" "+diagnosis)
    fig.savefig(filename+".pdf")
    plt.clf()  # Plots clash without cleaning the buffer, which is not done automatically if not put on screen


while True:
    database_yn = input("Dump database(y/n): ")
    plot_yn = input("Dump plots as pdf(y/n): ")
    if database_yn == 'y' or plot_yn == 'y':
        break

contents = os.listdir()
contents.sort()
pathlist = []
for i in contents:
    if i[-3:] == 'wav':
        pathlist.append(i)

with open("patient_diagnosis.csv") as database:
    patient_diagnosis = {i.split(",")[0]:i.split(",")[1][0:-1] for i in database.readlines()}

fh = open("skipped_data.txt", 'w')
if database_yn == 'y':
    txt_database = open("database_new.csv", 'w')
    txt_index = open("index_codes.txt", 'w')

counter = 0
for filename in pathlist:
    print("Processing:", filename)
    wav, sr = librosa.load(filename)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
    if D.shape[1] < 432:
        fh.write(filename+" "+patient_diagnosis.get(filename[0:3])+'\n')
        continue
    D = D[6:186, 0:432]  # First 10 seconds (0:432), btw 60-2000 Hz (6:186)

    if plot_yn == 'y':
        plotter(D, filename, patient_diagnosis.get(filename[0:3]))
    if database_yn != 'y':
        continue  # Skips database formation if user not opted

    # Info on diagnoses put above the stft output:
    txt_database.write(f"{counter}, ")  # Index for data
    txt_index.write(f"{counter}: {filename}\n")
    counter += 1

    patinf = patient_diagnosis.get(filename[0:3])
    if patinf == "Healthy":
        txt_database.write("1, ")
    else:
        txt_database.write("0, ")
    if patinf == "COPD":
        txt_database.write("1, ")
    else:
        txt_database.write("0, ")
    if patinf == "Pneumonia" or patinf == "Pneumoni":
        txt_database.write("1, ")
    else:
        txt_database.write("0, ")
    if patinf == "URTI":
        txt_database.write("1, ")
    else:
        txt_database.write("0, ")
    if patinf == "Bronchiectasis":
        txt_database.write("1, ")
    else:
        txt_database.write("0, ")
    if patinf == "Bronchiolitis":
        txt_database.write("1\n")
    else:
        txt_database.write("0\n")

    # Following loop calculates mean of 3x3 squares in D, dumps the resulting list to database
    for i in range(60):
        processed_array = [np.mean(D[3*i:3*i+3, 3*j:3*j+3]) for j in range(144)]
        txt_database.write(f"{str(processed_array)[1:-1]}\n")

if database_yn == 'y':
    txt_database.close()  # First line of entry for data of file with index i, starting from line 0: 61*(i-1)

print("Done processing all available data")
