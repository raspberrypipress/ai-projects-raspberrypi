import pandas as pd
import psutil
import os
import re
import time
from datetime import datetime

# Collect some CPU activity/thermal data
# Intuitively, we all know these are related.
# But can some of these predict the others? Reliably?

def get_cpu_speed():
    val = os.popen("vcgencmd measure_clock arm").read()
    return float(val[val.find("=")+1:]) / 10**9


def get_cpu_temp():
    val = os.popen("vcgencmd measure_temp").read()
    return float(re.search(r"temp=([\d\.]+)", val)[1])


def get_volts():
    val = os.popen("vcgencmd measure_volts").read()
    return float(re.search(r"volt=([\d\.]+)", val)[1])

numsamples = 2500
samples = []
for i in trange(numsamples):
    samples.append([get_cpu_speed(), get_volts(), psutil.cpu_percent(), psutil.getloadavg()[0], get_cpu_temp()])
    time.sleep(0.25)

df = pd.DataFrame(samples)
df.columns = ['cpu_speed', 'volts', 'cpu_pct', 'load_avg', 'temp']
df.to_csv(f"readings-{datetime.today().strftime('%Y-%m-%d-%H.00')}.csv", index=False)