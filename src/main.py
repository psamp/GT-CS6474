import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from model1 import n_grams
from model2 import activity_reputation
from model3 import narratives

# read in and split dataset into test/train
data = pd.read_json('./data/pizza_request_dataset.json')

y = data['requester_received_pizza'].astype(int).tolist()
X_train, X_test, y_train, y_true = train_test_split(data, y, test_size=0.1)

# print("MODEL 1 - N-GRAMS")
# n_grams(X_train, X_test, y_train, y_true)
# print('\n')
# print("MODEL 2 - ACTIVITY & REPUTATION")
# activity_reputation(X_train, X_test, y_train, y_true)
# print('\n')
print("MODEL 3 - NARRATIVES")
narratives(X_train, X_test, y_train, y_true)