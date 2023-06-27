import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from os import killpg
from typing_extensions import TypeVarTuple
import numpy as np
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os

# Define parameters for further use
input_dimension = 7
num_hex_files = 99
num_cyl_files = 90
buffer_length = 50

# Preprocess the hexagon data
for k in range(1, num_hex_files+1):
    file_path = f'hexagon{k}.npy'
    matrix = np.load(file_path)
    # Delete the first five rows of the matrix, until the sensor readings stabilize
    matrix = np.delete(matrix, (0, 1, 2, 3, 4, 5), axis=0)
    dimensions = matrix.shape
    continue_loop = True
    # Deal with null values. If a null value appears in the first 100 rows, delete the matrix. If after, truncate matrix.
    for i in range(0, dimensions[0]):
        for j in range(0, input_dimension):
            if np.isnan(matrix[i, j]):
                if i > 100:
                    print(
                        f"Null value found in {file_path}, position ({i}, {j}), and was modified to exclude the null value.")
                    matrix = matrix[0:i, :]
                    np.save(file_path, matrix)
                    continue_loop = False
                    break
                else:
                    try:
                        os.remove(f"hexagon{k}.npy")
                        print(f"hexagon{k}.npy was removed succesfully.")
                    except:
                        print(
                            f"Null value found in {file_path}, position ({i}, {j}), but couldn't be modified to fit requirements. Please manually delete the file.")
                sleep(.2)
        if not continue_loop:
            break
    if continue_loop:
        print(f"{file_path} has been cleared of NaN values.")

    np.save(file_path, matrix)

hex_list = []
for k in range(1, num_hex_files+1):
    if os.path.exists(f"hexagon{k}.npy"):
        hex_list.append(k)
num_hex_files = len(hex_list)
for i in range(1, num_hex_files):
    if i not in hex_list:
        matrix = np.load(f"hexagon{max(hex_list)}.npy")
        np.save(f"hexagon{i}.npy")
        os.remove(f"hexagon{max(hex_list)}.npy")
        hex_list.append(i)
        hex_list.remove(max(hex_list))

# Preprocess the cylinder data
for k in range(1, num_cyl_files+1):
    file_path = f'cylinder{k}.npy'
    matrix = np.load(file_path)
    # Delete the first five rows of the matrix, until the sensor readings stabilize
    # matrix = np.delete(matrix, (0, 1, 2, 3, 4, 5), axis=0)
    dimensions = matrix.shape
    continue_loop = True
    # Deal with null values. If a null value appears in the first 100 rows, delete the matrix. If after, truncate matrix.
    for i in range(0, dimensions[0]):
        for j in range(0, input_dimension):
            if np.isnan(matrix[i, j]):
                if i > 100:
                    print(
                        f"Null value found in {file_path}, position ({i}, {j}), and was modified to exclude the null value.")
                    matrix = matrix[0:i, :]
                    np.save(file_path, matrix)
                    continue_loop = False
                    break
                else:
                    try:
                        os.remove(f"cylinder{k}.npy")
                        print(f"cylinder{k}.npy was removed succesfully.")
                    except:
                        print(
                            f"Null value found in {file_path}, position ({i}, {j}), but couldn't be modified to fit requirements. Please manually delete the file.")
                sleep(.2)
        if not continue_loop:
            break
    if continue_loop:
        print(f"{file_path} has been cleared of NaN values.")

    np.save(file_path, matrix)

cyl_list = []
for k in range(1, num_cyl_files+1):
    if os.path.exists(f"cylinder{k}.npy"):
        cyl_list.append(k)
num_cyl_files = len(cyl_list)
for i in range(1, num_cyl_files):
    if i not in cyl_list:
        matrix = np.load(f"cylinder{max(cyl_list)}.npy")
        np.save(f"cylinder{i}.npy")
        os.remove(f"cylinder{max(cyl_list)}.npy")
        cyl_list.append(i)
        cyl_list.remove(max(cyl_list))


# Load the data
cylinder_data = [np.load(f'cylinder{i}.npy') for i in range(1, num_cyl_files)]
hexagon_data = [np.load(f'hexagon{i}.npy') for i in range(1, num_hex_files)]

# Combine the data and create corresponding labels
X = np.concatenate((cylinder_data, hexagon_data))
y = np.concatenate((np.zeros(len(cylinder_data)), np.ones(len(hexagon_data))))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

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

stride = 1
X_train_subsequences = []
y_train_subsequences = []
X_test_subsequences = []
y_test_subsequences = []

for i in range(len(X_train_normalized)):
    num_subsequences = (
        len(X_train_normalized[i]) - buffer_length) // stride + 1
    for j in range(num_subsequences):
        start_index = j * stride
        end_index = start_index + buffer_length
        X_train_subsequences.append(
            X_train_normalized[i][start_index:end_index])
        y_train_subsequences.append(y_train[i])

for i in range(len(X_test_normalized)):
    num_subsequences = (
        len(X_test_normalized[i]) - buffer_length) // stride + 1
    for j in range(num_subsequences):
        start_index = j * stride
        end_index = start_index + buffer_length
        X_test_subsequences.append(X_test_normalized[i][start_index:end_index])
        y_test_subsequences.append(y_test[i])

X_train_subsequences = np.array(X_train_subsequences)
y_train_subsequences = np.array(y_train_subsequences)
X_test_subsequences = np.array(X_test_subsequences)
y_test_subsequences = np.array(y_test_subsequences)


# Build and train the LSTM model with Early Stopping
model = Sequential()
model.add(LSTM(64, input_shape=(buffer_length,
          X_train_subsequences.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss', patience=1, restore_best_weights=True)

# Fit the model with EarlyStopping callback
history = model.fit(X_train_subsequences, y_train_subsequences,
                    epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_subsequences, y_test_subsequences)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Plot the training loss versus validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


