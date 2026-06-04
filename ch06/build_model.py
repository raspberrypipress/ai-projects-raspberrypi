import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import emlearn

current_window = []
X_set = []
y_set = []
window_size = 10
rolling_flick = []
data = []
multiplier=1000

def to_int(data_list):
    output = []
    for datum in data_list:
        output.append(int(float(datum) * multiplier))
    return output

def rolling_av(data, index):
    total = 0
    for item in range(index - window_size, index):
        total = total + int(data[item][4])
     
    if total / window_size < 0.5:
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
    X_set.append(to_int(predictor))
    y_set.append(rolling_av(data, index))

X_train, X_test, y_train, y_test = train_test_split(X_set,
                y_set, test_size=0.2, random_state=42)
estimator = RandomForestClassifier(n_estimators=20, 
                max_depth=15, max_features=7, random_state=42)
estimator.fit(X_train, y_train)

score = estimator.score(X_test, y_test)
print("score is: ", score)

# Convert model using emlearn
cmodel = emlearn.convert(estimator, method='inline')

# Save as loadable .csv file
path = 'flick_model.csv'
cmodel.save(file=path, name='flick', format='csv')
print('Wrote model to', path)
