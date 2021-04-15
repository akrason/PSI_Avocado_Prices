import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import sklearn.model_selection as ms

df = pd.read_csv("avocado.csv")
df.describe()
# rename learning column
df = df.rename(columns={"AveragePrice": "target"})

# corelations
abs(df.corr()).sort_values(by="target")

# generate histogram
df["target"].hist()
# generate hist data
df["target"].value_counts()
df = df.drop("Date", axis = 1)
df = df.drop("type", axis = 1)
df = df.drop("region", axis = 1)
df = df.drop("Lp", axis = 1)
# matrix without target column AKA input data
X = df.drop("target",axis =1)


# output data
y = df["target"]

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.50, random_state=42)

# check if hist for test data is simillar to original one
y_test.hist()

# generate test histogram data
y_test.value_counts()

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(500, activation="relu"),  # numOfNeurons,activationFunction
    Dense(500, activation="relu"),
    Dense(250, activation="relu"),
    Dense(250, activation="relu"),
    Dense(125, activation="relu"),
    Dense(125, activation="relu"),
    Dense(1)
])
# network parameters
model.compile(
    loss="mse",
    optimizer="adam",
    metrics=["MAPE"])

# sumary of network model
model.summary()

# start learning
# epochs is numOfIterations
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

# show accuracy throughout learning proccess
plt.plot(history.history["val_MAPE"])

np.max(history.history["val_MAPE"])
