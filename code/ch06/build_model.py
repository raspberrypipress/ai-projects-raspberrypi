import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
import emlearn

current_window = []
windows = []
window_res = []
window_size = 10
rolling_flick = []
data = []
multiplier=1000

def to_int(data_list):
    output = []
    for datum in data_list:
        output.append(int(float(datum)*multiplier))
    return output

def rolling_av(data, index):
    total = 0
    for item in range(index-window_size, index):
        total = total + int(data[item][4])
     
    if total/window_size < 0.5:
        return 1
    return 0

with open('wand_data.csv') as csvfile:
    flickreader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in flickreader:
        data.append(row)
     
for index in range(10, len(data)):
    predictor = []
    for y in range(0, 3):
        for x in range(-window_size, 0):
            predictor.append(data[index + x][y + 1])
    windows.append(to_int(predictor))
    window_res.append(rolling_av(data, index))

estimator = RandomForestClassifier(n_estimators=20, 
                max_depth=15, max_features=7, random_state=1)
estimator.fit(windows[0:800], window_res[0:800])

score = get_scorer('f1')(estimator, windows, window_res)
print("score is: ",score)

# Convert model using emlearn
cmodel = emlearn.convert(estimator, method='inline')

# Save as loadable .csv file
path = 'flick_model.csv'
cmodel.save(file=path, name='flick', format='csv')
print('Wrote model to', path)