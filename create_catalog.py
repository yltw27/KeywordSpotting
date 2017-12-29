import os
import csv
from random import shuffle
from datetime import datetime


def collect_wav(path, val_percent):
    start = datetime.now()
    wav_files = []
    count_0, count_1 = 0, 0
    
    for dirpaths, dirnames, filenames in os.walk(path):
        for f in filenames:
            if '.wav' in f:
                label = f[-5]
                if label == '0':
                    count_0 += 1
                    wav_files.append((os.path.join(dirpaths, f), 1, 0))
                elif label == '1':
                    count_1 += 1
                    wav_files.append((os.path.join(dirpaths, f), 0, 1))
                else:
                    print('{} without label'.format((os.path.join(dirpaths, f))))
    print('data: {}'.format(len(wav_files)))
    
    shuffle(wav_files)
    sep_point = int(len(wav_files) * (1-val_percent))
    training_data = wav_files[:sep_point]
    validation_data = wav_files[sep_point:]
    print('training data: {} validation data: {}'.format(len(training_data), len(validation_data)))
    
    with open('./data/train.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['wav_files', 'label0', 'label1']
        writer.writerow(header)
        writer.writerows(training_data)
        
    with open('./data/validation.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['wav_files', 'label0', 'label1']
        writer.writerow(header)
        writer.writerows(validation_data)
        
    end = datetime.now()
    total = count_0 + count_1
    print('label_0 : {} ({:.3f} %), label_1 : {} ({:.3f} %)'.format(count_0, count_0/total*100, count_1, count_1/total*100))
    print('Create training and validation catalogs in ./data\nDuration: {}\n'.format(str(end-start)))

if __name__ == '__main__':
    collect_wav(path='./data/', val_percent=0.2)

    
