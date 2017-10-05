from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
import pandas as pd
np.random.seed(7)
import data_process as dp 

def getData(self, filename):
    filename = filename
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    data = list(reader)
    data = np.array(data).astype('float')
    df = pd.DataFrame(data)
    return df

