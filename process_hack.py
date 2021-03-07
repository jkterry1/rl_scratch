import psutil
import schedule
import time

"""
pids = psutil.pids()

new_ghosts = []

for pid in pids:
    p = psutil.Process(pid)
    if p.name() == 'ray::ImplicitFunc':
        if p.memory_percent() > 0.0:
            if p.cpu_percent(interval=1.0) == 0.0:  # slow check
                new_ghosts.append(pid)

new_ghosts = set(new_ghosts)

try:
    old_ghosts = pickle.load(open('/home/justin_terry/ghosts.pkl', 'rb'))
except:
    old_ghosts = set()

to_kill = old_ghosts.intersection(new_ghosts)

for kill in to_kill:
    p = psutil.Process(kill)
    p.kill()

new_old_ghosts = new_ghosts - old_ghosts

pickle.dump(new_old_ghosts, open('/home/justin_terry/ghosts.pkl', 'wb'))
"""


def killer():
    pids = psutil.pids()

    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == 'ray::ImplicitFunc':
            if p.memory_percent() > 0.0:
                if p.cpu_percent(interval=1.0) == 0.0:  # slow check
                    p.kill()


schedule.every(5).seconds.do(killer)

while True:
    schedule.run_pending()
    time.sleep(1)
