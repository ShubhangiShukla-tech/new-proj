from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocess import load_data

X, y = load_data()

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(30, 30, 1)),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

model.save('wafer_model.h5')