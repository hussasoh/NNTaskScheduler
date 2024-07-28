import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime, timedelta


# Generate realistic training data
def generate_realistic_data(num_samples=1000):
    np.random.seed(42)
    tasks = np.arange(1, num_samples + 1)
    deadlines = np.random.randint(1, 30, num_samples)  # deadlines within a month
    priorities = np.random.randint(1, 5, num_samples)  # priorities from 1 (low) to 4 (high)
    time_needed = np.random.randint(1, 10, num_samples)  # time needed in hours per task

    data = pd.DataFrame({
        'tasks': tasks,
        'deadlines': deadlines,
        'priorities': priorities,
        'time_needed': time_needed
    })

    # Simulate time slots based on some heuristic
    data['time_slots'] = (data['deadlines'] / data['time_needed']) * data['priorities']
    data['time_slots'] = data['time_slots'].apply(lambda x: max(1, min(24, int(x))))  # Cap to 24 hours

    return data


realistic_data = generate_realistic_data()


# Data Preparation
def prepare_data(data):
    features = data[['tasks', 'deadlines', 'priorities', 'time_needed']]
    target = data['time_slots']

    # Normalize the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)

    # Normalize the target
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    scaled_data = np.hstack((scaled_features, scaled_target))

    return scaled_data, feature_scaler, target_scaler


# Prepare data
scaled_data, feature_scaler, target_scaler = prepare_data(realistic_data)

# Split the data into training and testing sets
train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Model Design
model = Sequential()
model.add(Dense(64, input_dim=train_data.shape[1] - 1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Training the model
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Prediction function
def predict_task_time_slot(new_task):
    scaled_new_task = feature_scaler.transform(new_task)
    predicted_scaled_time_slot = model.predict(scaled_new_task)
    predicted_time_slot = target_scaler.inverse_transform(predicted_scaled_time_slot)
    return predicted_time_slot


def format_time_slot(hours):
    whole_hours = int(hours)
    minutes = int((hours - whole_hours) * 60)
    return f"{whole_hours} hours and {minutes} minutes"


# Scheduling function
def generate_schedule(tasks, daily_availability):
    total_minutes_per_day = daily_availability * 60  # Convert hours to minutes
    schedule = []
    current_date = datetime.now()
    daily_minutes_used = 0

    for task in tasks:
        task_id, deadline, priority, time_needed, predicted_time_slot = task
        duration_minutes = int(predicted_time_slot * 60)
        time_needed_minutes = int(time_needed * 60)  # Convert time needed to minutes

        while duration_minutes > 0:
            if daily_minutes_used + time_needed_minutes > total_minutes_per_day:
                # Move to the next day if we exceed daily availability
                daily_minutes_used = 0
                current_date += timedelta(days=1)

            start_time = daily_minutes_used
            end_time = daily_minutes_used + min(time_needed_minutes, total_minutes_per_day - daily_minutes_used)
            if end_time > total_minutes_per_day:
                end_time = total_minutes_per_day

            schedule.append((task_id, current_date, start_time, end_time))
            daily_minutes_used += end_time - start_time

            duration_minutes -= time_needed_minutes
            if duration_minutes > 0:
                # If there's still time needed, start a new day
                daily_minutes_used = 0
                current_date += timedelta(days=1)

    return schedule


# User Interface using Tkinter
def add_task():
    try:
        deadline = int(deadline_entry.get())
        priority = int(priority_combobox.get())
        time_needed = int(time_needed_entry.get())

        if any([deadline <= 0, priority <= 0, time_needed <= 0]):
            raise ValueError("Inputs must be positive integers.")

        new_task_id = len(tasks) + 1
        new_task = np.array([[new_task_id, deadline, priority, time_needed]])
        predicted_time_slot = predict_task_time_slot(new_task)[0][0]
        tasks.append((new_task_id, deadline, priority, time_needed, predicted_time_slot))
        tasks_listbox.insert(tk.END,
                             f"Task {new_task_id}: Time Needed: {format_time_slot(time_needed)} | Predicted Time Slot: {format_time_slot(predicted_time_slot)}")

        deadline_entry.delete(0, tk.END)
        priority_combobox.set("")
        time_needed_entry.delete(0, tk.END)
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


def generate_daily_schedule():
    try:
        daily_availability = int(daily_availability_entry.get())
        if daily_availability <= 0:
            raise ValueError("Daily availability must be a positive integer.")

        schedule = generate_schedule(tasks, daily_availability)
        schedule_text = ""
        for task_id, date, start_time, end_time in schedule:
            start_hour, start_minute = divmod(start_time, 60)
            end_hour, end_minute = divmod(end_time, 60)
            schedule_text += (f"Task {task_id} on {date.strftime('%Y-%m-%d')}: {start_hour:02}:{start_minute:02} - "
                              f"{end_hour:02}:{end_minute:02}\n")

        show_schedule_window(schedule_text)
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


def show_schedule_window(schedule_text):
    schedule_window = tk.Toplevel(root)
    schedule_window.title("Daily Schedule")
    schedule_text_box = tk.Text(schedule_window, wrap=tk.WORD, width=80, height=20)
    schedule_text_box.insert(tk.END, schedule_text)
    schedule_text_box.config(state=tk.DISABLED)
    schedule_text_box.pack(expand=True, fill=tk.BOTH)
    tk.Button(schedule_window, text='Close', command=schedule_window.destroy).pack(pady=10)


# Tkinter UI setup
root = tk.Tk()
root.title("Smart Activity Planner")

tasks = []

tk.Label(root, text="Deadline (days)").grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
tk.Label(root, text="Priority").grid(row=1, column=0, padx=15, pady=10, sticky=tk.W)
tk.Label(root, text="Time Needed for Task (hours)").grid(row=2, column=0, padx=15, pady=10, sticky=tk.W)
tk.Label(root, text="Daily Availability (hours)").grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)

deadline_entry = tk.Entry(root, width=35)
priority_combobox = ttk.Combobox(root, values=[1, 2, 3, 4], width=33)
time_needed_entry = tk.Entry(root, width=35)
daily_availability_entry = tk.Entry(root, width=35)

deadline_entry.grid(row=0, column=1, padx=15, pady=10)
priority_combobox.grid(row=1, column=1, padx=15, pady=10)
time_needed_entry.grid(row=2, column=1, padx=15, pady=10)
daily_availability_entry.grid(row=3, column=1, padx=15, pady=10)

tasks_listbox = tk.Listbox(root, width=80, height=15)
tasks_listbox.grid(row=0, column=2, rowspan=4, padx=15, pady=10)

tk.Button(root, text='Add Task', command=add_task).grid(row=4, column=0, sticky=tk.W, pady=15, padx=15)
tk.Button(root, text='Generate Schedule', command=generate_daily_schedule).grid(row=4, column=1, sticky=tk.E, pady=15,
                                                                                padx=15)
tk.Button(root, text='Quit', command=root.quit).grid(row=4, column=2, sticky=tk.E, pady=15, padx=15)

root.mainloop()
