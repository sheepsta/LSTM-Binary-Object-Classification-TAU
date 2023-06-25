import numpy as np
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Check for nan values in the inputted numpy matrices

for i in range(1, 49):
  file_path = f'hexagon{i}.npy'
  matrix = np.load(file_path)
  dimensions = matrix.shape
  for i in range(0, dimensions[0]):
    for j in range(0, 7):
      if np.isnan(matrix[i, j]):
        print(f"Found it!!!!! Null value in {file_path}, position ({i}, {j})")
        sleep(.2)

for i in range(1, 51):
  file_path = f'cylinder{i}.npy'
  matrix = np.load(file_path)
  dimensions = matrix.shape
  for i in range(0, dimensions[0]):
    for j in range(0, 7):
      if np.isnan(matrix[i, j]):
        print(f"Found it!!!!! Null value in {file_path}, position ({i}, {j})")
        sleep(.2)

# Load the data
cylinder_data = [np.load(f'cylinder{i}.npy') for i in range(1, 51)]
hexagon_data = [np.load(f'hexagon{i}.npy') for i in range(1, 49)]

# Combine the data and create corresponding labels
X = np.concatenate((cylinder_data, hexagon_data))
y = np.concatenate((np.zeros(len(cylinder_data)), np.ones(len(hexagon_data))))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()

X_train_normalized = []
for sequence in X_train:
    sequence_normalized = scaler.fit_transform(sequence)
    X_train_normalized.append(sequence_normalized)
X_train_normalized = np.array(X_train_normalized)

X_test_normalized = []
for sequence in X_test:
    sequence_normalized = scaler.transform(sequence)
    X_test_normalized.append(sequence_normalized)
X_test_normalized = np.array(X_test_normalized)



# Process the data into subsequences of length 18
buffer_length = 50
stride = 1
X_train_subsequences = []
y_train_subsequences = []
X_test_subsequences = []
y_test_subsequences = []

for i in range(len(X_train_normalized)):
    num_subsequences = (len(X_train_normalized[i]) - buffer_length) // stride + 1
    for j in range(num_subsequences):
        start_index = j * stride
        end_index = start_index + buffer_length
        X_train_subsequences.append(X_train_normalized[i][start_index:end_index])
        y_train_subsequences.append(y_train[i])

for i in range(len(X_test_normalized)):
    num_subsequences = (len(X_test_normalized[i]) - buffer_length) // stride + 1
    for j in range(num_subsequences):
        start_index = j * stride
        end_index = start_index + buffer_length
        X_test_subsequences.append(X_test_normalized[i][start_index:end_index])
        y_test_subsequences.append(y_test[i])

X_train_subsequences = np.array(X_train_subsequences)
y_train_subsequences = np.array(y_train_subsequences)
X_test_subsequences = np.array(X_test_subsequences)
y_test_subsequences = np.array(y_test_subsequences)

from tensorflow.keras.callbacks import EarlyStopping

# Build and train the LSTM model with Early Stopping
model = Sequential()
model.add(LSTM(64, input_shape=(buffer_length, X_train_subsequences.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model with EarlyStopping callback
model.fit(X_train_subsequences, y_train_subsequences, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_subsequences, y_test_subsequences)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


