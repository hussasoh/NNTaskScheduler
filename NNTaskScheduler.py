import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime, timedelta
from tkcalendar import DateEntry
import csv
import os

# Constants
TASKS_CSV = 'tasks.csv'
AVAILABILITY_CSV = 'availability.csv'
PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Initialize daily_availability
daily_availability = {}
tasks = []

# Generate realistic training data
def generate_realistic_data(num_samples=1000):
    np.random.seed(42)
    tasks = np.arange(1, num_samples + 1)
    deadlines = np.random.randint(1, 30, num_samples)
    priorities = np.random.randint(1, 5, num_samples)
    time_needed = np.random.randint(1, 10, num_samples)

    data = pd.DataFrame({
        'tasks': tasks,
        'deadlines': deadlines,
        'priorities': priorities,
        'time_needed': time_needed
    })

    data['time_slots'] = (data['deadlines'] / data['time_needed']) * data['priorities']
    data['time_slots'] = data['time_slots'].apply(lambda x: max(1, min(24, int(x))))

    return data

# Data Preparation
def prepare_data(data):
    features = data[['tasks', 'deadlines', 'priorities', 'time_needed']]
    target = data['time_slots']

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    scaled_data = np.hstack((scaled_features, scaled_target))

    return scaled_data, feature_scaler, target_scaler

# Prepare and split data
realistic_data = generate_realistic_data()
scaled_data, feature_scaler, target_scaler = prepare_data(realistic_data)
train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Build and train the model
def build_and_train_model(train_data):
    model = Sequential()
    model.add(Dense(64, input_dim=train_data.shape[1] - 1, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    return model

model = build_and_train_model(train_data)

def predict_task_time_slot(new_task):
    scaled_new_task = feature_scaler.transform(new_task)
    predicted_scaled_time_slot = model.predict(scaled_new_task)
    predicted_time_slot = target_scaler.inverse_transform(predicted_scaled_time_slot)
    return predicted_time_slot

def format_time_slot(hours):
    whole_hours = int(hours)
    minutes = int((hours - whole_hours) * 60)
    return f"{whole_hours} hours and {minutes} minutes"

def generate_schedule(tasks, daily_availability):
    schedule = []
    current_date = datetime.now().date()
    daily_minutes_used = 0

    for task in tasks:
        task_id, deadline, priority, time_needed, predicted_time_slot, description = task
        time_needed_minutes = int(time_needed * 60)

        while time_needed_minutes > 0:
            day_of_week = current_date.strftime('%A')
            total_minutes_per_day = daily_availability.get(day_of_week, 0) * 60

            if daily_minutes_used + time_needed_minutes > total_minutes_per_day:
                daily_minutes_used = 0
                current_date += timedelta(days=1)
                day_of_week = current_date.strftime('%A')
                total_minutes_per_day = daily_availability.get(day_of_week, 0) * 60

            start_time = daily_minutes_used
            end_time = daily_minutes_used + min(time_needed_minutes, total_minutes_per_day - daily_minutes_used)
            if end_time > total_minutes_per_day:
                end_time = total_minutes_per_day

            schedule.append((task_id, current_date, start_time, end_time, description))
            daily_minutes_used += end_time - start_time

            time_needed_minutes -= (end_time - start_time)
            if time_needed_minutes > 0:
                daily_minutes_used = 0
                current_date += timedelta(days=1)

    return schedule

def add_task():
    try:
        deadline_date = deadline_entry.get_date()
        deadline = (deadline_date - datetime.now().date()).days
        priority_text = priority_combobox.get()
        priority = PRIORITY_MAP[priority_text]
        time_needed = int(time_needed_entry.get())
        description = description_entry.get()

        if deadline <= 0 or priority not in range(1, 5) or time_needed <= 0 or not description:
            raise ValueError("Invalid input.")

        new_task_id = len(tasks) + 1
        new_task = np.array([[new_task_id, deadline, priority, time_needed]])
        predicted_time_slot = predict_task_time_slot(new_task)[0][0]
        tasks.append((new_task_id, deadline, priority, time_needed, predicted_time_slot, description))
        tasks_listbox.insert(tk.END, f"Task {new_task_id}: {description} | Time Needed: {format_time_slot(time_needed)} | Predicted Time Slot: {format_time_slot(predicted_time_slot)}")
        save_tasks_to_file()

        deadline_entry.set_date(datetime.now())
        priority_combobox.set("")
        time_needed_entry.delete(0, tk.END)
        description_entry.delete(0, tk.END)
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def open_modify_window(selected_task):
    modify_window = tk.Toplevel(root)
    modify_window.title("Modify Task")

    tk.Label(modify_window, text="Description").grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
    tk.Label(modify_window, text="Deadline").grid(row=1, column=0, padx=15, pady=10, sticky=tk.W)
    tk.Label(modify_window, text="Priority").grid(row=2, column=0, padx=15, pady=10, sticky=tk.W)
    tk.Label(modify_window, text="Time Needed for Task (hours)").grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)

    mod_description_entry = tk.Entry(modify_window, width=35)
    mod_deadline_entry = DateEntry(modify_window, width=33, background='darkblue', foreground='white', borderwidth=2)
    mod_priority_combobox = ttk.Combobox(modify_window, values=list(PRIORITY_MAP.keys()), width=33)
    mod_time_needed_entry = tk.Entry(modify_window, width=35)

    task_id, deadline, priority, time_needed, predicted_time_slot, description = selected_task

    mod_description_entry.grid(row=0, column=1, padx=15, pady=10)
    mod_description_entry.insert(0, description)
    mod_deadline_entry.grid(row=1, column=1, padx=15, pady=10)
    mod_deadline_entry.set_date(datetime.now() + timedelta(days=deadline))
    mod_priority_combobox.grid(row=2, column=1, padx=15, pady=10)
    mod_priority_combobox.set(list(PRIORITY_MAP.keys())[priority-1])
    mod_time_needed_entry.grid(row=3, column=1, padx=15, pady=10)
    mod_time_needed_entry.insert(0, str(time_needed))

    def update_task():
        try:
            new_deadline_date = mod_deadline_entry.get_date()
            new_deadline = (new_deadline_date - datetime.now().date()).days
            new_priority_text = mod_priority_combobox.get()
            new_priority = PRIORITY_MAP[new_priority_text]
            new_time_needed = int(mod_time_needed_entry.get())
            new_description = mod_description_entry.get()

            if new_deadline <= 0 or new_priority not in range(1, 5) or new_time_needed <= 0 or not new_description:
                raise ValueError("Invalid input.")

            new_task = np.array([[task_id, new_deadline, new_priority, new_time_needed]])
            new_predicted_time_slot = predict_task_time_slot(new_task)[0][0]
            tasks[index] = (task_id, new_deadline, new_priority, new_time_needed, new_predicted_time_slot, new_description)
            tasks_listbox.delete(index)
            tasks_listbox.insert(index, f"Task {task_id}: {new_description} | Time Needed: {format_time_slot(new_time_needed)} | Predicted Time Slot: {format_time_slot(new_predicted_time_slot)}")
            save_tasks_to_file()
            modify_window.destroy()
        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    tk.Button(modify_window, text='Update', command=update_task).grid(row=5, column=0, columnspan=2, pady=15)

def modify_task():
    try:
        selected_task_index = tasks_listbox.curselection()
        if not selected_task_index:
            raise ValueError("No task selected.")
        global index
        index = selected_task_index[0]
        selected_task = tasks[index]

        open_modify_window(selected_task)
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def remove_task():
    try:
        selected_task_index = tasks_listbox.curselection()
        if not selected_task_index:
            raise ValueError("No task selected.")
        index = selected_task_index[0]
        tasks.pop(index)
        tasks_listbox.delete(index)
        save_tasks_to_file()
    except ValueError as ve:
        messagebox.showerror("Input Error", f"Invalid input: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def generate_daily_schedule():
    try:
        if not daily_availability:
            raise ValueError("Daily availability must be specified.")
        schedule = generate_schedule(tasks, daily_availability)

        grouped_tasks = {}
        for task_id, date, start_time, end_time, description in schedule:
            day_name = date.strftime('%A')
            start_hour, start_minute = divmod(start_time, 60)
            end_hour, end_minute = divmod(end_time, 60)
            if task_id not in grouped_tasks:
                grouped_tasks[task_id] = {'description': description, 'slots': []}
            grouped_tasks[task_id]['slots'].append(f"{day_name} ({date}): {start_hour:02}:{start_minute:02} - {end_hour:02}:{end_minute:02}")

        schedule_text = ""
        for task_id, details in grouped_tasks.items():
            schedule_text += f"**Task {task_id}:** {details['description']}\n"
            for slot in details['slots']:
                schedule_text += f"   **Time Slot:** {slot}\n"
            schedule_text += "\n"

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

def save_tasks_to_file():
    with open(TASKS_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Task ID', 'Deadline', 'Priority', 'Time Needed', 'Predicted Time Slot', 'Description'])
        for task in tasks:
            writer.writerow(task)

def load_tasks_from_file():
    if os.path.exists(TASKS_CSV):
        with open(TASKS_CSV, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                task_id, deadline, priority, time_needed, predicted_time_slot, description = row
                tasks.append((int(task_id), int(deadline), int(priority), int(time_needed), float(predicted_time_slot), description))
                tasks_listbox.insert(tk.END, f"Task {task_id}: {description} | Time Needed: {format_time_slot(float(time_needed))} | Predicted Time Slot: {format_time_slot(float(predicted_time_slot))}")

def save_availability_to_file():
    with open(AVAILABILITY_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'Availability'])
        for day, hours in daily_availability.items():
            writer.writerow([day, hours])

def load_availability_from_file():
    global daily_availability
    if os.path.exists(AVAILABILITY_CSV):
        with open(AVAILABILITY_CSV, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                day, hours = row
                daily_availability[day] = int(hours)

def open_availability_window():
    availability_window = tk.Toplevel(root)
    availability_window.title("Set Weekly Availability")

    tk.Label(availability_window, text="Day of the Week", font=("Helvetica", 12, "bold")).grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
    tk.Label(availability_window, text="Availability (Hours)", font=("Helvetica", 12, "bold")).grid(row=0, column=1, padx=15, pady=10, sticky=tk.W)

    availability_entries = {}
    for i, day in enumerate(DAYS_OF_WEEK):
        tk.Label(availability_window, text=day, font=("Helvetica", 10)).grid(row=i+1, column=0, padx=15, pady=10, sticky=tk.W)
        entry = tk.Entry(availability_window, width=35)
        entry.grid(row=i+1, column=1, padx=15, pady=10)
        if day in daily_availability:
            entry.insert(0, daily_availability[day])
        availability_entries[day] = entry

    def save_availability():
        global daily_availability
        try:
            daily_availability = {}
            for day, entry in availability_entries.items():
                value = entry.get()
                if value == '':
                    raise ValueError(f"Availability for {day} must be specified.")
                daily_availability[day] = int(value)
                if daily_availability[day] <= 0:
                    raise ValueError(f"Availability for {day} must be a positive integer.")
            save_availability_to_file()
            availability_window.destroy()
        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    tk.Button(availability_window, text='Save', command=save_availability).grid(row=8, column=0, columnspan=2, pady=15)

# Tkinter UI setup
root = tk.Tk()
root.title("Smart Activity Planner")

tk.Label(root, text="Description").grid(row=0, column=0, padx=15, pady=10, sticky=tk.W)
tk.Label(root, text="Deadline").grid(row=1, column=0, padx=15, pady=10, sticky=tk.W)
tk.Label(root, text="Priority").grid(row=2, column=0, padx=15, pady=10, sticky=tk.W)
tk.Label(root, text="Time Needed for Task (hours)").grid(row=3, column=0, padx=15, pady=10, sticky=tk.W)

description_entry = tk.Entry(root, width=35)
deadline_entry = DateEntry(root, width=33, background='darkblue', foreground='white', borderwidth=2)
priority_combobox = ttk.Combobox(root, values=list(PRIORITY_MAP.keys()), width=33)
time_needed_entry = tk.Entry(root, width=35)

description_entry.grid(row=0, column=1, padx=15, pady=10)
deadline_entry.grid(row=1, column=1, padx=15, pady=10)
priority_combobox.grid(row=2, column=1, padx=15, pady=10)
time_needed_entry.grid(row=3, column=1, padx=15, pady=10)

listbox_frame = tk.Frame(root)
listbox_frame.grid(row=0, column=2, rowspan=5, padx=15, pady=10, sticky='nsew')

tasks_listbox = tk.Listbox(listbox_frame, width=80, height=15)
tasks_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tasks_scrollbar_y = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
tasks_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

tasks_scrollbar_x = tk.Scrollbar(listbox_frame, orient=tk.HORIZONTAL)
tasks_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

tasks_listbox.config(yscrollcommand=tasks_scrollbar_y.set, xscrollcommand=tasks_scrollbar_x.set)
tasks_scrollbar_y.config(command=tasks_listbox.yview)
tasks_scrollbar_x.config(command=tasks_listbox.xview)

tk.Button(root, text='Add Task', command=add_task).grid(row=5, column=0, sticky=tk.W, pady=15, padx=15)
tk.Button(root, text='Set Availability', command=open_availability_window).grid(row=5, column=1, sticky=tk.W, pady=15, padx=15)
tk.Button(root, text='Generate Schedule', command=generate_daily_schedule).grid(row=5, column=1, sticky=tk.E, pady=15, padx=15)
tk.Button(root, text='Quit', command=root.quit).grid(row=5, column=2, sticky=tk.E, pady=15, padx=15)

def show_context_menu(event):
    context_menu.tk_popup(event.x_root, event.y_root)

context_menu = tk.Menu(root, tearoff=0)
context_menu.add_command(label="Modify Task", command=modify_task)
context_menu.add_command(label="Remove Task", command=remove_task)

tasks_listbox.bind("<Button-3>", show_context_menu)

load_tasks_from_file()
load_availability_from_file()
root.mainloop()
