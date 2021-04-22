import pandas as pd
from sklearn.model_selection import train_test_split
from models import *

# read in and split dataset into test/train
data = pd.read_json('/Users/psampson/Documents/code/social-computing/assignment-3/data/pizza_request_dataset.json')
train, test = train_test_split(data, test_size=0.1)

n_grams(test)


