from datetime import datetime
from time import sleep
import os


def recording(mode, people, path, start):
    start_point = datetime.now()
    print("Start {}'s {} recording(s)... (mode = {})".format(people, start+1, mode))
    
    filename = datetime.now().strftime("%Y%m%d_") + people + '_' + '{:04d}'.format(start) + '_' + str(mode) + ".wav"
    
    os.system("arecord -d 3 -r 16000 -f S16_LE " + path + filename)
    
    print('Recording duration: ', datetime.now() - start_point)
    print('=' * 60)


def input_test():
    people = input("Enter your initials: (e.g. CL, CY, BL...) ")
    mode = int(input("Enter the mode: (1:with keyword, 0:without keyword)"))
    times = int(input("Enter recording times: "))    
    print('Recording will start in 3 seconds.')
    for i in range(3):
        print(3-i)
        sleep(1)
    return people, mode, times


if __name__ == '__main__':
    people, mode, times = input_test()

    try:
        os.mkdir("data/" + people)
    except OSError:
        pass

    for i in range(times):
        recording(path='data/'+people+'/', mode=mode, people=people, start=i)


