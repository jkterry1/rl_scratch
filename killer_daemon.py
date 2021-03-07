import schedule
import time
import os


def killer():
    os.system('python3 /home/justin_terry/all_pistonball/killer.py')


schedule.every(3).seconds.do(killer)

while True:
    schedule.run_pending()
    time.sleep(1)
