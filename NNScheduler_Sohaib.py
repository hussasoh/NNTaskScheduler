import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Function to get user input
def get_user_input():
    tasks = []
    num_tasks = int(input("Enter the number of tasks: "))

    for i in range(num_tasks):
        task_id = i + 1
        task_name = input(f"Enter name for task {task_id}: ")
        deadline = input(f"Enter deadline for task {task_id} (YYYY-MM-DD): ")
        priority = int(input(f"Enter priority for task {task_id} (1-5): "))
        availability = float(input(f"Enter your availability for task {task_id} (0-1): "))
        tasks.append({'task_id': task_id, 'task_name': task_name, 'deadline': deadline, 'priority': priority,
                      'user_availability': availability})

    df = pd.DataFrame(tasks)
    return df


# Data preparation for LSTM
def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['user_availability']].values)

    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    # Dynamic time step adjustment
    time_step = min(3, len(df) - 1)
    if time_step < 1:
        print("Not enough data to create a dataset. Please enter more tasks.")
        return None, None, None

    X, Y = create_dataset(scaled_data, time_step)

    if X.size == 0 or Y.size == 0:
        print("Insufficient data for the given time step. Please enter more tasks or reduce the time step.")
        return None, None, None

    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, Y, scaler


# Build and train LSTM model
def train_lstm_model(X, Y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=100, batch_size=1, verbose=1)
    return model


# Task scheduling based on predicted availability
def schedule_tasks(model, df, scaler):
    scaled_availability = scaler.transform(df[['user_availability']].values)
    predicted_availability = model.predict(scaled_availability)
    df['predicted_availability'] = scaler.inverse_transform(predicted_availability)
    df.sort_values(by=['priority', 'predicted_availability'], ascending=[False, True], inplace=True)
    return df


# Main execution
df = get_user_input()
X, Y, scaler = prepare_data(df)

if X is not None and Y is not None:
    print("Data prepared successfully.")
    lstm_model = train_lstm_model(X, Y)
    print("LSTM model trained successfully.")
    scheduled_df = schedule_tasks(lstm_model, df, scaler)
    print("Tasks scheduled successfully.")
    print(scheduled_df)
else:
    print("Failed to prepare data. Please check the input and try again.")
