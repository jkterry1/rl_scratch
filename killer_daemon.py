import schedule
import time
import os
from pathlib import Path


def killer():
    os.system('python3 ' + str(Path.home()) + '/rl_scratch/killer.py')


schedule.every(3).seconds.do(killer)

while True:
    schedule.run_pending()
    time.sleep(1)
