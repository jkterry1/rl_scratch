import schedule
import time
import os
from pathlib import Path


class render:
    def __init__(self):
        self.running_policy_list = []
        # self.max_concurrent_renders

    def render(self):
        policy_list = os.listdir(Path.home()+'/policies')
        new_policies = set(self.running_policy_list)-set(policy_list)
        self.running_policy_list = policy_list

        for policy in new_policies:
            os.system('python3 /home/justin_terry/rl_scratch/render.py '+Path.home()+'/'+policy)


render_class = render()

schedule.every(60).seconds.do(render_class.render())

while True:
    schedule.run_pending()
    time.sleep(1)
