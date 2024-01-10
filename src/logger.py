import os
import pandas as pd
from time import time
from datetime import timedelta


class Logger:
    def __init__(self, log_dir, log_name, verbose=True):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_str_filepath = os.path.join(log_dir, log_name+'.log')
        self.log_lst_filepath = os.path.join(log_dir, log_name+'.dat')
        self.log_s = ''
        self.log_l = []
        self.verbose = verbose
        self.t0 = time()
        self.ts = self.t0

    def store(self, **kwargs):
        self.log_l.append(kwargs)
        s = pd.Series(kwargs)
        with open(self.log_lst_filepath, 'a') as fs:
            fs.write(s.to_json()+'\n')

    def print_profiling_info(self, n_curr, n_step, n_total):
        dt = time() - self.t0
        dts = time() - self.ts
        eta = (n_total-n_curr)*dt/n_step
        self.print("> Elapsed time: {}".format(timedelta(seconds=dt)))
        self.print("> Since last call: {}".format(timedelta(seconds=dts)))
        self.print("> ETA: {}".format(timedelta(seconds=eta)))
        self.ts = time()

    def restart_timer(self):
        self.ts = time()

    def print(self, line_raw):
        line = str(line_raw)
        self.log_s += line + '\n'
        with open(self.log_str_filepath, 'a') as fs:
            fs.write(line + '\n')
        if self.verbose:
            print(line)