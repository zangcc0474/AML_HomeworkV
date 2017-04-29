from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import keras
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.grid_search import GridSearchCV
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

num_classes = 3
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def make_model(optimizer="adam", hidden_size=32):
    model = Sequential([
        Dense(32, input_shape=(4,)),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(3),
        Activation('softmax'),
    ])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
    return model


clf = KerasClassifier(make_model)
param_grid = {'epochs': [1, 5, 10],  # epochs is fit parameter, not in make_model!
              'hidden_size': [32, 64, 256]}
grid = GridSearchCV(clf, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)

res = pd.DataFrame(grid.cv_results_)
res.pivot_table(index=["param_epochs", "param_hidden_size"],
                values=['mean_train_score', "mean_test_score"])

print("Test Accuracy:")
print (np.mean((grid.predict(X_test) == y_test)))



