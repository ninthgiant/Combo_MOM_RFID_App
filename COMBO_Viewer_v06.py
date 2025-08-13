################
#
#   PIT_Viewer_v03 by RAMauck, July 18 2024 with help from ChatGPT
#       Allows user to choose a PIT tag file and quickly find bird activity 
#       Once file opens, a list of days in the file is shown in the Days popup
#       User choose a day to display
#           All records from that day show in middle window
#           All birds found in that day are summarized in teh right window
#
###########

##########################
#   Import libraries needed
#       tkinter for interface
#       pandas for database management
########
import tkinter as tk
from tkinter import messagebox, filedialog, Menu, simpledialog, Scrollbar
import pandas as pd
import os  # Import the os module for file operations


##########################
#   Define globally available variables
########

# Define dataframe globally
dataframe = pd.DataFrame() 

# universal date time format
date_fmt = '%m-%d-%Y %H:%M:%S'

# universal day only format
day_only_fmt = '%m-%d-%Y'

global vVersString
global vAppName
vVersString = " (v_05)"  ## upDATE AS NEEDED - v04 July 18 - add first and last readers enocuntered
vAppName = "PIT Viewer" + vVersString

#############
# return_useful_name: takes a path string and returns just the name of the file
####
def return_useful_name(the_path):
    where = the_path.rfind("/")
    the_name = the_path[(where + 1):(len(the_path)-4)]
    return the_name

##########################
#   function: load_file
#       Mac interface to open a file
#       Once opened, put in dataframe and display
#       For now, it builds global dataframe, but could return the dataframe instead - for later?
########
def load_file():
    global dataframe
    
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
    if file_path:
        try:
            filename = os.path.basename(file_path)
            if filename.startswith('RFID_'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    if file_path.endswith('.csv') or file_path.endswith('.txt') or file_path.endswith('.TXT'):
                        # Read CSV without assuming headers
                        dataframe = pd.read_csv(file, delimiter=',', header=None, names=['PIT_ID', 'Rdr', 'PIT_DateTime'], 
                                                on_bad_lines='warn')

                        # Ensure 'PIT_DateTime' is converted to datetime if necessary
                        if 'PIT_DateTime' in dataframe.columns:
                            dataframe['PIT_DateTime'] = pd.to_datetime(dataframe['PIT_DateTime'], errors='coerce')
                        else:
                            raise ValueError("'PIT_DateTime' column not found in the dataframe.")

                    else:
                        raise ValueError("Unsupported file format. Please select a CSV or text file.")
                    
                    # Display dataframe in t1
                    t1.delete('1.0', tk.END)
                    t1.insert(tk.END, dataframe.to_string(index=False))

                    # make sure t2 and t3 are empty now
                    t2.delete('1.0', tk.END)
                    t3.delete('1.0', tk.END)
                    
                    # Update Days menu with unique days from PIT_DateTime
                    update_days_menu()

                    # Update the label showing the number of records
                    record_count = len(dataframe)
                    # and the name of the file for the user to see
                    label_all_records.config(text=f"All Records {filename} ({record_count})")
                
            else:
                raise ValueError("Selected file does not start with 'RFID_'.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")


##########################
#   function: insert_dataframe_into_widget
#       One function to update windows with dataframe information
#       Parameters:
#           - dataframe (pd.DataFrame): The DataFrame containing data to be inserted.
#           - widget (tk.Text): The tkinter Text widget to insert data into
########
def insert_dataframe_into_widget(dataframe, widget):

    for index, row in dataframe.iterrows():
        widget.insert(tk.END, f"{row['PIT_ID']}    {row['Rdr']}    {row['PIT_DateTime']}\n")

##########################
#   function: update_days_menu
#       Gets list of unique days from the dataframe
#       Builds a list of those days and puts them in a popup widget
#       Handles the user choosing a day from the popup by calling show_records_by_day 
########
def update_days_menu():
    global dataframe
    
    try:
        
        if dataframe.empty or 'PIT_DateTime' not in dataframe.columns:
            print("DataFrame is empty or 'PIT_DateTime' column not found.")
            return
        
        # Convert 'PIT_DateTime' to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(dataframe['PIT_DateTime']):
            dataframe['PIT_DateTime'] = pd.to_datetime(dataframe['PIT_DateTime'], errors='coerce')
        
        # Extract unique days from 'PIT_DateTime', handling NaT values
        valid_dates = dataframe['PIT_DateTime'][pd.notnull(dataframe['PIT_DateTime'])].dt.date.unique()
        # sort the dates
        unique_days = sorted(valid_dates, reverse=True)
        
        # Clear existing menu items
        days_menu.delete(0, tk.END)
        
        # Add days to the menu
        for day in unique_days:
            formatted_day = day.strftime('%m-%d-%Y')
            days_menu.add_command(label=formatted_day, command=lambda d=day: show_records_by_day(d))
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update Days menu: {e}")

##########################
#   function: update_Unique_Tags_menu
#       Gets list of unique days from the dataframe
#       Builds a list of those days and puts them in a popup widget
#       Handles the user choosing a day from the popup by calling show_records_by_day 
########
def update_Unique_Tags_menu():
    global dataframe
    
    my_Tags = show_pit_tags()

    try:
        
        if my_Tags.empty:
            print("DataFrame is empty or 'PIT_DateTime' column not found.")
            return
        
        
        # Clear existing menu items
        days_menu.delete(0, tk.END)

        # Clear existing menu items
        pit_menu.delete(0, tk.END)

        print("now ad pit to mb_pit")
        
        # Add days to the menu
       # for pit in my_Tags:
        #    pit_menu.add_command(label='Unique RFID', command=lambda p=pit: show_records_by_pit(p))

        for pit_id in my_Tags['PIT_ID']:
                pit_menu.add_command(label=str(pit_id), command=lambda p=pit_id: show_records_by_PIT(p))


        print("done ad pit to mb_pit")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to update Days menu: {e}")

##########################
#   function: show_pit_tags
#       Gets list of unique tags from the dataframe
#       Builds a list of those tags and puts them in window t2
#       NOT USED as of 7/18/2024, but may be used in the future 
########
def show_pit_tags():
    try:
        unique_pit_tags = dataframe[['PIT_ID']].drop_duplicates()
        
        # Clear existing content in t2
        t2.delete('1.0', tk.END)
        
        # Insert unique PIT Tags into t2
        for index, row in unique_pit_tags.iterrows():
            t2.insert(tk.END, f"{row['PIT_ID']}\n")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to retrieve PIT Tags: {e}")

    return unique_pit_tags

##########################
#   function: show_records_by_PIT
#       Gets list of records for a particular night - 8PM of chosen day until 6AM of next day
#       Builds a list of those tags and puts them in window t2
#       Tallies all PIT records for each bird found during that window and when they were there 
########
def show_records_by_PIT(selected_pit):

    print("In show_records_by_pit")
    try:
        global dataframe
        
        # Print the dataframe with selected columns
        print("DataFrame before finding all for one PIT:")
        print(dataframe[['PIT_ID', 'Rdr']])  # Correct way to select multiple columns
        
        # Filter dataframe by selected PIT_ID
        filtered_df = dataframe[dataframe['PIT_ID'] == selected_pit]
        
        print("Get Selected records:")
        print(filtered_df)
     
        
        # Clear existing content in t2 and t3
        # t2.delete('1.0', tk.END)
        t3.delete('1.0', tk.END)

        ## update the label
        record_count = len(filtered_df)
        
        print("Good to here - after record count")

        label_one_day.config(text=f"Records for {selected_pit} ({record_count} records)")

        insert_dataframe_into_widget(filtered_df, t3)

        print("Good to here - after insert to widget")
        
        # Display earliest and latest datetime for each PIT ID in t3
        for pit_id in filtered_df['PIT_ID'].unique():
            pit_subset = filtered_df[filtered_df['PIT_ID'] == pit_id]
            earliest_datetime = pit_subset['PIT_DateTime'].min()
            latest_datetime = pit_subset['PIT_DateTime'].max()
            # get the readers that read the earliest and latest
            # first_rdr = pit_subset['Rdr'].iloc[0]  # Value in the first row
            # last_rdr = pit_subset['Rdr'].iloc[-1]  # Value in the last row

            # Format datetime into %m-%d-%Y format
            formatted_earliest = earliest_datetime.strftime(date_fmt)
            formatted_latest = latest_datetime.strftime(date_fmt)

            # Count records for current PIT ID
            record_count = len(pit_subset)

            t3.insert(tk.END, f"PIT ID: {pit_id}\n")
            #t3.insert(tk.END, f"\tFirst: \t{formatted_earliest}\tRdr: {first_rdr}\n")
            #t3.insert(tk.END, f"\tLast: \t{formatted_latest}\tRdr: {last_rdr}\n")
            t3.insert(tk.END, f"\tRecords Count: \t{record_count}\n\n")
            print("Good to here - each line")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to retrieve records: {e}")


##########################
#   function: show_records_by_day
#       Gets list of records for a particular night - 8PM of chosen day until 6AM of next day
#       Builds a list of those tags and puts them in window t2
#       Tallies all PIT records for each bird found during that window and when they were there 
########
def show_records_by_day(selected_day):
    try:
        global dataframe
        
        # Convert selected_day to datetime object
        selected_date = pd.to_datetime(selected_day)
        
        # Define start and end times for the selected day
        start_time = selected_date.replace(hour=20, minute=0, second=0)
        end_time = selected_date + pd.Timedelta(days=1, hours=6, minutes=0, seconds=0)
        
        # Filter dataframe by selected day
        filtered_df = dataframe[
            (dataframe['PIT_DateTime'] >= start_time) & 
            (dataframe['PIT_DateTime'] <= end_time)
        ]
        
        # Clear existing content in t2 and t3
        t2.delete('1.0', tk.END)
        t3.delete('1.0', tk.END)

        ## update the label
        record_count = len(filtered_df)
        
        formatted_date = selected_day.strftime("%m-%d-%Y")
        # label_one_day.config(text=f"Night of {formatted_date}")
        label_one_day.config(text=f"Night of {formatted_date} ({record_count} records)")

        insert_dataframe_into_widget(filtered_df, t2)
        
        # Display earliest and latest datetime for each PIT ID in t3
        for pit_id in filtered_df['PIT_ID'].unique():
            pit_subset = filtered_df[filtered_df['PIT_ID'] == pit_id]
            earliest_datetime = pit_subset['PIT_DateTime'].min()
            latest_datetime = pit_subset['PIT_DateTime'].max()
            # get the readers that read the earliest and latest
            first_rdr = pit_subset['Rdr'].iloc[0]  # Value in the first row
            last_rdr = pit_subset['Rdr'].iloc[-1]  # Value in the last row

            # Format datetime into %m-%d-%Y format
            formatted_earliest = earliest_datetime.strftime(date_fmt)
            formatted_latest = latest_datetime.strftime(date_fmt)

            # Count records for current PIT ID
            record_count = len(pit_subset)

            t3.insert(tk.END, f"PIT ID: {pit_id}\n")
            t3.insert(tk.END, f"\tFirst: \t{formatted_earliest}\tRdr: {first_rdr}\n")
            t3.insert(tk.END, f"\tLast: \t{formatted_latest}\tRdr: {last_rdr}\n")
            t3.insert(tk.END, f"\tRecords Count: \t{record_count}\n\n")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to retrieve records: {e}")


##########################
#   Setup GUI
#       All things tKinter related
#       Currently has:
#           One button to open a file
#           One popup menu to list days covered in the file
#           3 windows to display data
#           One menu (File) with one item (Quit)
########

##########################
#   What do do when you quit the app
#######
def quit_app():
    root.quit()

##########################
# Create the main application windowp
#######
root = tk.Tk()
root.title(vAppName)

##########################
# Create the main application window - buttons and widgets
#######

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate window size (80% of screen width)
window_width = int(screen_width * 0.8)
window_height = screen_height

root.geometry(f"{window_width}x{window_height}")

# Create a frame to hold the buttons and date menu
frame_buttons_date = tk.Frame(root)
frame_buttons_date.pack(side=tk.TOP, fill=tk.X, padx=20, pady=20)

# Create buttons in the frame
button_load_file = tk.Button(frame_buttons_date, text="Load File", command=load_file)
button_load_file.pack(side=tk.LEFT)

# Create buttons in the frame for Show individuals - added v06
button_load_file = tk.Button(frame_buttons_date, text="Birds", command=update_Unique_Tags_menu)
button_load_file.pack(side=tk.LEFT)

# Create a Menubutton for Days
mb_days = tk.Menubutton(frame_buttons_date, text="Days", indicatoron=True, borderwidth=1, relief="raised")
mb_days.pack(side=tk.LEFT, padx=10)
days_menu = tk.Menu(mb_days, tearoff=0)
mb_days.configure(menu=days_menu)

# Create a Menubutton for RFID/PIT
mb_pit = tk.Menubutton(frame_buttons_date, text="RFID", indicatoron=True, borderwidth=1, relief="raised")
mb_pit.pack(side=tk.LEFT, padx=10)
pit_menu = tk.Menu(mb_pit, tearoff=0)
mb_pit.configure(menu=pit_menu)

# NOT USED NOW - maybe later if we want to focus on one PIT tag and do something with it
# button_pit_tags = tk.Button(frame_buttons_date, text="PIT Tags", command=show_pit_tags)
# button_pit_tags.pack(side=tk.LEFT, padx=10)

# button_date = tk.Button(frame_buttons_date, text="Date", command=show_records_by_date)
# button_date.pack(side=tk.LEFT, padx=10)

# Create the output frame with a border
outputFrame = tk.Frame(root, width=50, height=500, bd=1, relief=tk.SOLID)
outputFrame.pack(pady=5)

# Create labels for output frames and grid them
label_all_records = tk.Label(outputFrame, text="All Records", font=("Arial", 12))
label_all_records.grid(row=0, column=0, padx=10, pady=(10, 5))

label_one_day = tk.Label(outputFrame, text="One day", font=("Arial", 12))
label_one_day.grid(row=0, column=1, padx=10, pady=(10, 5))

label_bird_activity = tk.Label(outputFrame, text="Bird Activity", font=("Arial", 12))
label_bird_activity.grid(row=0, column=2, padx=10, pady=(10, 5))

# Create frames for each text widget
frame_t1 = tk.Frame(outputFrame, bd=1, relief=tk.SOLID)
frame_t1.grid(row=1, column=0, padx=10, pady=5, sticky=tk.NSEW)

frame_t2 = tk.Frame(outputFrame, bd=1, relief=tk.SOLID)
frame_t2.grid(row=1, column=1, padx=10, pady=5, sticky=tk.NSEW)

frame_t3 = tk.Frame(outputFrame, bd=1, relief=tk.SOLID)
frame_t3.grid(row=1, column=2, padx=10, pady=5, sticky=tk.NSEW)

# Create scrollbars for each text widget
scrollbar1 = Scrollbar(frame_t1, orient="vertical")
scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar2 = Scrollbar(frame_t2, orient="vertical")
scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar3 = Scrollbar(frame_t3, orient="vertical")
scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)

# Create text widgets in their respective frames - Controls the height of the text boxes
myOutputWidth = 50
myOutputHeight = 50  # Adjust as needed

t1 = tk.Text(frame_t1, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar1.set)
t1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar1.config(command=t1.yview)

t2 = tk.Text(frame_t2, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar2.set)
t2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar2.config(command=t2.yview)

t3 = tk.Text(frame_t3, width=myOutputWidth, height=myOutputHeight)
t3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Configure scrollbars to scroll with text widgets
t1.config(yscrollcommand=scrollbar1.set)
t2.config(yscrollcommand=scrollbar2.set)

##########################
# Create the main application window - menus
#######
# Create a menu bar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Create a File menu with Quit option
file_menu = tk.Menu(menubar, tearoff=False)
menubar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Quit", command=quit_app)

##########################
# Start the app going wiht the mainloop
#######
root.mainloop()
