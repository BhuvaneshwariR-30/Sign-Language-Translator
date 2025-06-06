import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Reshape data to be compatible with CNN input
# Assume each element in data is a set of landmarks, each landmark has x, y coordinates
# If each landmark is a (x, y) pair, reshape the data as (num_samples, num_landmarks, 2)
data_reshaped = []
for d in data:
    # Reshape the data to (num_landmarks, 2) - as each landmark has 2 values (x, y)
    reshaped_landmarks = np.array(d)
    data_reshaped.append(reshaped_landmarks)

# Convert to numpy array
data_reshaped = np.array(data_reshaped)

# Add a channel dimension (CNNs expect data to have 4 dimensions: samples, height, width, channels)
# Here we have 2 dimensions for each landmark (x, y) and we treat each sample as an image with 2 channels (x and y)
data_reshaped = data_reshaped[..., np.newaxis]  # Add channel dimension (making it (num_samples, num_landmarks, 2, 1))

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_reshaped, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))  # First convolutional layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer

model.add(Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer

model.add(Conv2D(128, (3, 3), activation='relu'))  # Third convolutional layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer

# Flatten the data for fully connected layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output layer: assuming a classification task with multiple classes
model.add(Dense(len(np.unique(labels)), activation='softmax'))  # Change softmax to sigmoid for binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)  # Convert predictions to class labels

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
model.save('cnn_model.h5')
