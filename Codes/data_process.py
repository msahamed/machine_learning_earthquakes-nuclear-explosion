import csv
import pandas as pd

class dataProcess:
    def getDataFromVtk(self):
    
    def getData(self, filename):
        filename = filename
        raw_data = open(filename, 'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        data = list(reader)
        data = np.array(data).astype('float')
        df = pd.DataFrame(data)
        return df

    def processData(self):
        scalex = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        X = scalex.fit_transform(X)
        scaley = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        Y = scaley.fit_transform(Y)
        return df

    def splitData(self, X, Y, test_size):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        return X, Y