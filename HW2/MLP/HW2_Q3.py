from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

x = np.loadtxt('data.csv', delimiter=',')
y = np.loadtxt('labels.csv', delimiter=',')

# normalize data
x_normed = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

x_train, x_test, y_train, y_test = train_test_split(x_normed, y, test_size=0.25, random_state=42)

# creat model with 4 hidden layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(213, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(213, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(150, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# model.compile(optimizer='Adagrad', loss='mean_squared_error', metrics=['accuracy'])

# set the input and number of epochs 
model.fit(x_train, y_train, epochs=40)

# calculate accuracy using test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print('loss value and accuracy is: ')
print(val_loss, val_acc)
