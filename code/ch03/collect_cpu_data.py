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
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"No number found in string: {s}")

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
    samples.append([get_cpu_speed(), get_volts(), psutil.cpu_percent(), psutil.getloadavg()[0], get_cpu_temp()])
    time.sleep(0.25)
    pbar.set_description(f"{samples[-1][0]:.2f}GHz {samples[-1][1]:.2f}V {samples[-1][2]:.1f}% {samples[-1][3]:.2f} {samples[-1][4]:.2f}C ")

df = pd.DataFrame(samples)
df.columns = ['cpu_speed', 'volts', 'cpu_pct', 'load_avg', 'temp']
df.to_csv(f"readings-{datetime.today().strftime('%Y-%m-%d-%H.00')}.csv", index=False)
