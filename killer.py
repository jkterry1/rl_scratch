import psutil

pids = psutil.pids()

for pid in pids:
    p = psutil.Process(pid)
    if p.name() == 'ray::ImplicitFunc':
        if p.memory_percent() > 0.0:
            if p.cpu_percent(interval=1.0) == 0.0:  # slow check
                p.kill()
