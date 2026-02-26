import pandas as pd
import psutil
import os # for os.popen to run vcgencmd commands
import re # for regex to extract numbers from vcgencmd output
import time # for time.sleep to control sampling rate
from tqdm import trange # for progress bar

# Collect some CPU activity/thermal data
# Intuitively, we all know these are related.
# But can some of these predict the others? Reliably?

# vcgencmd uses a format like "temp=45.2'C", so we need to
# extract the numbers with decimal points if any.
def extract_number(s):
    match = re.search(r"=(\d+(\.\d+)?)", s)
    return round(float(match.group(1)), 4)

def speed():
    val = os.popen("vcgencmd measure_clock arm").read()
    return extract_number(val) / 10**9

def temp():
    val = os.popen("vcgencmd measure_temp").read()
    return extract_number(val)

def volts():
    val = os.popen("vcgencmd measure_volts").read()
    return extract_number(val)

if __name__ == "__main__":
    numsamples = 2500
    samples = []
    pbar = trange(numsamples)

    for i in pbar:
        cpu_pct = psutil.cpu_percent()
        load_avg = round(psutil.getloadavg()[0], 2)
        r = [speed(), volts(), cpu_pct, load_avg, temp()]

        samples.append(r)
        pbar.set_description(
            f"{r[0]}GHz {r[1]}V {r[2]}% {r[3]} {r[4]}C")
        time.sleep(0.25)

    df = pd.DataFrame(samples)
    df.columns = ['cpu_speed', 'volts', 'cpu_pct',
                  'load_avg', 'temp']
    df.to_csv(f"readings.csv", index=False)