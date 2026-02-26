import pandas as pd
import psutil
import os
import re
import time
from datetime import datetime
from tqdm import trange

# Collect some CPU activity/thermal data
# Intuitively, we all know these are related.
# But can some of these predict the others? Reliably?

# vgencmd returns results in a format like "frequency(0)=1500019456",
# "volt=0.8898V", or "temp=45.2'C" so we need to extract the numbers.
def extract_number(s):
    match = re.search(r"=(\d+(\.\d+)?)", s)
    return round(float(match.group(1)), 4)

def get_cpu_speed():
    val = os.popen("vcgencmd measure_clock arm").read()
    return extract_number(val) / 10**9

def get_cpu_temp():
    val = os.popen("vcgencmd measure_temp").read()
    return extract_number(val)

def get_volts():
    val = os.popen("vcgencmd measure_volts").read()
    return extract_number(val)

numsamples = 2500
samples = []
pbar = trange(numsamples)
for i in pbar:
    r = [get_cpu_speed(), get_volts(), psutil.cpu_percent(), psutil.getloadavg()[0], get_cpu_temp()]
    samples.append(r)
    time.sleep(0.25)
    pbar.set_description(f"{r[0]}GHz {r[1]}V {r[2]}% {r[3]} {r[4]}C ")

df = pd.DataFrame(samples)
df.columns = ['cpu_speed', 'volts', 'cpu_pct', 'load_avg', 'temp']
df.to_csv(f"readings-{datetime.today().strftime('%Y-%m-%d-%H.00')}.csv", index=False)
