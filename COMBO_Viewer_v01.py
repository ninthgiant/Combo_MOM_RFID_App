################
#
#   Combo_Viewer_v01 by RAMauck, Aug 2025 with help from ChatGPT
#       Allows user to link MOM data with RFID data via the timestamps
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
import numpy as np
import os  # Import the os module for file operations

##########################
#   Define globally available variables
########

# Define dataframe globally
dataframe = pd.DataFrame() 

# universal date time format
date_fmt = '%m/%d/%Y %H:%M:%S'

# universal day only format
day_only_fmt = '%m-%d-%Y'

global do_print
do_print = True  # Set to True to enable print statements for debugging
global vVersString
global vAppName
vVersString = " (v_01)"  ## upDATE AS NEEDED - v04 July 18 - add first and last readers enocuntered
vAppName = "Combo Viewer" + vVersString

########################### 
#   function: format_time_cols  
#       Takes a dataframe and formats all columns ending in _Time to the given date_fmt
#       If cols is given, only formats those columns
#       usage:out = format_time_cols(out, date_fmt, cols=["MOM_Time", "RFID_Time"])
##########
def format_time_cols(df, date_fmt, cols=None, as_string=True):
    if cols is None:
        cols = [c for c in df.columns if c.endswith("_Time")]
    for c in cols:
        s = pd.to_datetime(df[c], errors="coerce")
        df[c] = s.dt.strftime(date_fmt).fillna("") if as_string else s
    return df

###########################
#   function: filter_outside_hours
#       Filters a DataFrame to return rows where the hour of a specified datetime column
#       is outside a specified range (before hr_am or after hr_pm).
#       Parameters:
#           df (pd.DataFrame): The DataFrame to filter.
#           datetime_col (str): The name of the datetime column to check.
#           hr_am (int): The hour before which to filter (e.g., 7 for 7 AM).
#           hr_pm (int): The hour after which to filter (e.g., 20 for 8 PM).
#       Returns:
#           pd.DataFrame: A DataFrame containing only the rows where the hour is outside the specified range.
###########
def filter_outside_hours(df, datetime_col, hr_am, hr_pm):

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    valid_time = df[datetime_col].notna()
    hours_mask = valid_time & (
        (df[datetime_col].dt.hour < hr_am) |
        (df[datetime_col].dt.hour > hr_pm)
    )
    return df.loc[hours_mask].copy()




##########################
#   function: return_folder_path
#       Opens a dialog to select a folder and prints the path and files in it
#       If do_print is True, it will print the folder path and all files in it
#######
def return_folder_path():
    global do_print
    folder_path = filedialog.askdirectory()

    if do_print: 
        print(folder_path)
        all_files = os.listdir(folder_path)
    
        # Filter out only files (not directories)
        files = [f for f in all_files if os.path.isfile(os.path.join(folder_path, f))]

        # Print each file name
        for file_name in files: print(file_name)

#############
# return_useful_name: takes a path string and returns just the name of the file
####
def return_useful_name(the_path):
    where = the_path.rfind("/")
    the_name = the_path[(where + 1):(len(the_path)-4)]
    return the_name

#############
# extract_burrow: takes a filename string and returns the burrow code
####
def extract_burrow(val):
    s = str(val)

    # Remove file extension if present (.txt, .csv, any case)
    for ext in (".txt", ".csv"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break

    # Take the last 3 characters
    s = s[-3:]

    # Handle case like "5_3"
    if len(s) >= 2 and s[1] == "_":
        return s[2:].zfill(3)  # pad to 3 digits
    return s.replace("_", "").zfill(3)


 #############
 #       --- Clean Burrow formatting ---
 ##
def clean_burrow2(val: str) -> str:
    s = str(val)
    if len(s) >= 2 and s[1] == "_":  # e.g., '5_3'
        return s[2:]  # drop first two chars ('5' and '_')
    return s.replace("_", "")  # remove all underscores

def clean_burrow(val: str) -> str:
    s = str(val)

    # Remove file extension if present (.txt, .csv, any case)
    for ext in (".txt", ".csv"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break

    # Handle case like "5_3"
    if len(s) >= 2 and s[1] == "_":  # e.g., '5_3'
        s = s[2:]
    else:
        s = s.replace("_", "")

    # Return only if it's digits
    return s.zfill(3) if s.isdigit() else ""


def load_one_MOM_file():
    global dataframe
    
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])

    if file_path:
        try:
            filename = return_useful_name(file_path)  # Get just the name of the file without path and extension
            if filename.startswith('Bird_Weight_'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    if file_path.endswith('.csv') or file_path.endswith('.txt') or file_path.endswith('.TXT'):
                        # Read CSV without assuming headers
                        dataframe = pd.read_csv(file, delimiter=',', header=0, on_bad_lines='warn')

                        # Ensure 'DateTime' is converted to datetime if necessary
                        if 'DateTime' in dataframe.columns:
                            dataframe['DateTime'] = pd.to_datetime(dataframe['DateTime'], errors='coerce')
                        else:
                            raise ValueError("'DateTime' column not found in the dataframe.")

                    else:
                        raise ValueError("Unsupported file format. Please select a CSV or text file.")
                    
                    # populate_mom_Windows(dataframe) # moved this code to this function 7/18/2024 - can use it with one file or many files
                    populate_mom_Windows(dataframe[['DateTime', 'Burrow',  'Wt']]) # moved this code to this function 7/18/2024 - can use it with one file or many files
                    


                
            else:
                raise ValueError("Selected file does not start with 'Bird_Weight_'.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

##########################
#   function: get_All_MOM_data
#       Mac interface to open all MOM files in a folder - once used this folder: str
#       Once opened, put in dataframe and display
#       For now, it builds global dataframe, but could return the dataframe instead - for later?
########
def get_All_MOM_data(
    folder: None | str = None
    ) -> pd.DataFrame:

    if folder is None:
        folder = filedialog.askdirectory()
        if not folder:  # User cancelled the dialog
            print("No folder selected. Using default folder.")
    else:
        folder = "/Users/bobmauck/devel/Combo_App/Example_Data"
    
    cols_to_import = ['File', 'Trace_Segment_Num', 'DateTime', 'Wt_Min_Slope']

    try:
        files_in_folder = os.listdir(folder)
        for f in files_in_folder:
            pass # print(f"  {f}")
    except FileNotFoundError:
        print(f"Error: Folder not found: {folder}")
        return pd.DataFrame(columns=['MOM_File', 'Segment', 'DateTime', 'Wt_Min_Slope', 'Burrow'])

    all_dfs = []

       # Loop through matching files
    for filename in files_in_folder:
        if filename.lower().startswith("bird_weight_") and filename.lower().endswith(".txt"):
            file_path = os.path.join(folder, filename)
            try:
                df_temp = pd.read_csv(
                    file_path,
                    delimiter=',',
                    usecols=cols_to_import,
                    header=0,
                    on_bad_lines='warn'
                )
                # Extract burrow code from 'File' column
                df_temp['Burrow'] = df_temp['File'].astype(str).str[-7:-4]
                    # chg
                df_temp["Burrow"] = df_temp["Burrow"].astype(str).apply(clean_burrow)

                all_dfs.append(df_temp)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_dfs:
        print("No matching files found.")
        return pd.DataFrame(columns=['MOM_File', 'Segment', 'DateTime', 'Wt_Min_Slope', 'Burrow'])

    # Combine and drop duplicates
    df_MOM = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    # Rename columns
    df_MOM.rename(columns={'File': 'MOM_File', 'Trace_Segment_Num': 'Segment', 'Wt_Min_Slope': 'Wt'}, inplace=True)

    # CHG populate_mom_Windows(df_MOM[['DateTime', 'Burrow',  'Wt']]) # moved this code to this function 7/18/2024 - can use it with one file or many files

    df_MOM.sort_values(by=["Burrow", "DateTime"], inplace=True)


    return df_MOM

##########################
#   function: load_all_MOM_files
#       Calls get_All_MOM_data to get all MOM data in a folder
#       Puts it in the global dataframe
#       Then populates the windows with that data
########    
def load_all_MOM_files():
    global all_mom # use this to hold the dataframe for all MOM data for combo work?
    df_all_mom = get_All_MOM_data()


    # print(df_all_rfid.head(10))  # Print first 10 rows for verification
    # populate_mom_Windows(df_all_mom[['MOM_File', 'Segment', 'DateTime', 'Wt_Min_Slope', 'Burrow']]) # only if we have wider text window
    print("populate_mom_Windows")
    populate_mom_Windows(df_all_mom[['DateTime', 'Burrow', 'Wt']]) # moved this code to this function 7/18/2024 - can use it with one file or many files

    return df_all_mom 




##########################
#   function: load_file
#       Mac interface to open a single RFID file
#       Once opened, put in dataframe and display
#       For now, it builds global dataframe, but could return the dataframe instead - for later?
########
def load_file():
    global dataframe
    
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])

    if file_path:
        try:
            # filename = os.path.basename(file_path)
            filename = return_useful_name(file_path)  # Get just the name of the file without path and extension
            if filename.startswith('RF'):
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
                    

                    populate_RFID_Windows(dataframe) # moved this code to this function 7/18/2024 - can use it with one file or many files

                
            else:
                raise ValueError("Selected file does not start with 'RFID_'.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def clear_text_widgets(widget_dict):
    """
    Clears the content of all Text widgets in the given widget dictionary
    whose keys start with 't' (e.g., 't1', 'mom_t2', etc.).
    """
    for key, widget in widget_dict.items():
        if isinstance(widget, tk.Text) and key.lstrip("_").split("_")[-1].startswith("t"):
            widget.delete("1.0", tk.END)

def set_text_widget_font(output_frame_dict, font_name="Arial", font_size=14):
    """
    Set the font for all tk.Text widgets in the given output frame dictionary.

    :param output_frame_dict: Dictionary of widgets returned from create_output_frame()
    :param font_name: Font family name (default: Arial)
    :param font_size: Font size in points (default: 14)
    """
    for widget in output_frame_dict.values():
        if isinstance(widget, tk.Text):
            widget.config(font=(font_name, font_size))



##########################
#   function: populate_RFID_Windows
#       Takes a dataframe and puts it in the t1 window
#       Also updates the Days menu and Unique Tags menu
#       Updates the label showing the number of records
def populate_RFID_Windows(df_rfid):

    if df_rfid.empty:
        messagebox.showwarning("Warning", "DataFrame is empty. Please load data first.")
        return

    # Clear existing content in t1, t2, and t3
    clear_text_widgets(output_widgets)
    

    # Insert the entire dataframe into t1
    t1.insert(tk.END, df_rfid.to_string(index=False))

    # Update Days menu with unique days from PIT_DateTime
    # update_days_menu(df_rfid)

    # Update Unique Tags menu with unique PIT_IDs
    # update_Unique_Tags_menu(df_rfid)

    # Update the label showing the number of records
    record_count = len(df_rfid)
    # label_all_records.config(text=f"All Records ({record_count})")

##########################
#   function: populate_mom_Windows
#       Takes a dataframe and puts it in the t1 window
#       Also updates the Days menu and Unique Tags menu
#       Updates the label showing the number of records
def populate_mom_Windows(df_mom):

    if df_mom.empty:
        messagebox.showwarning("Warning", "DataFrame is empty. Please load data first.")
        return

    # Clear existing content in t1, t2, and t3, adjusts for # widgets
    clear_text_widgets(mom_widgets)

    # Insert the entire dataframe into t1
    mom_t1.insert(tk.END, df_mom.to_string(index=False))

    # Update Days menu with unique days from PIT_DateTime
    # update_days_menu(df_rfid)

    # Update Unique Tags menu with unique PIT_IDs
    # update_Unique_Tags_menu(df_rfid)

    # Update the label showing the number of records
    record_count = len(df_mom)
    # mom_label_all_records.config(text=f"All Records ({record_count})")

##########################
#   function: get_All_RFID_data
#       Mac interface to open all RFID files in a folder - once used this folder: str = "/Users/bobmauck/devel/Combo_App/Example_Data"
#       Once opened, put in dataframe and display
#           Load all files in 'folder' that start with 'RF' and end with '.txt',
#       combine them into a single DataFrame, remove duplicates, and return it.
#       Assumes the files have NO header row and uses:
#           ['PIT_ID', 'Rdr', 'PIT_DateTime'] as the column names.
########
def get_All_RFID_data(
    folder: None | str = None
    ) -> pd.DataFrame:

    if folder is None:
        folder = filedialog.askdirectory()
        if not folder:  # User cancelled the dialog
            print("No folder selected. Using default folder.")
    else:
        folder = "/Users/bobmauck/devel/Combo_App/Example_Data"
    
    cols_to_import = ['PIT_ID', 'Rdr', 'PIT_DateTime']

    # Debug info
    # print(f"Looking in folder: {folder}")
    try:
        files_in_folder = os.listdir(folder)
        # print("Files in folder:")
        for f in files_in_folder:
            #print(f"  {f}")
            pass  # Uncomment to print files in folder
    except FileNotFoundError:
        print(f"Error: Folder not found: {folder}")
        return pd.DataFrame(columns=cols_to_import)

    all_dfs = []

    # Loop through matching files
    for filename in files_in_folder:
        if filename.lower().startswith("rf") and filename.lower().endswith(".txt"):
            file_path = os.path.join(folder, filename)
            try:
                df_temp = pd.read_csv(
                    file_path,
                    delimiter=',',
                    header=None,                # No header in file
                    names=cols_to_import,       # Assign column names
                    usecols=[0, 1, 2],          # Only first 3 columns
                    on_bad_lines='warn'
                )
                # Add the filename column
                df_temp['RF_File'] = filename
                df_temp['Burrow'] = df_temp['RF_File'].astype(str).str[-7:-4]
                # df_temp["Burrow"] = df_temp["File"].apply(extract_burrow)

                    # chg
                df_temp["Burrow"] = df_temp["Burrow"].astype(str).apply(clean_burrow)
                
                all_dfs.append(df_temp)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_dfs:
        print("No matching files found.")
        return pd.DataFrame(columns=cols_to_import)

    # Combine and drop duplicates
    df_RFID = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    df_RFID.sort_values(by=["Burrow", "PIT_DateTime"], inplace=True)

    return df_RFID

##########################
#   function: load_all_RFID_files
#       Calls get_All_RFID_data to get all RFID data in a folder
#       Puts it in the global dataframe
#       Then populates the windows with that data
########    
def load_all_RFID_files():
    # global all_rfid # use this to hold the dataframe for all RFID data for combo work?
    df_all_rfid = get_All_RFID_data()
    df_all_rfid = format_time_cols(df_all_rfid, date_fmt, cols=['PIT_DateTime'])
    # chg   
    # df_all_rfid["Burrow"] = df_all_rfid["Burrow"].astype(str).apply(clean_burrow)

    populate_RFID_Windows(df_all_rfid[['PIT_DateTime', 'Burrow', 'Rdr', 'PIT_ID']])
    return df_all_rfid 

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
def update_days_menu(dataframe=None):
    # global dataframe
    
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
def update_Unique_Tags_menu(dataframe=None  ):
    #   global dataframe
    
    my_Tags = show_pit_tags(dataframe)

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
def show_pit_tags(dataframe=None):
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
def show_records_by_PIT(selected_pit, dataframe=None):

    print("In show_records_by_pit")
    try:
        # global dataframe
        
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

###############################
#   function: do_Join_MOM_RFID
#       Joins MOM data with RFID data based on timestamps.
#       function that calls join_MOM_RFID from button
########
def do_Join_MOM_RFID():
    # global all_mom, all_rfid
    messagebox.showwarning("Find Data", "Choose folder with rfid and mom data.")
    folder = filedialog.askdirectory()
    if not folder:  # User cancelled the dialog
        print("No folder selected. Using default folder.")

    all_mom = get_All_MOM_data(folder)
    all_rfid = get_All_RFID_data(folder)

    if all_mom.empty or all_rfid.empty:
        messagebox.showwarning("Warning", "Please load both MOM and RFID data before joining.")
        return

    try:
        # Call the join function
        joined_df = join_MOM_RFID2(all_mom, all_rfid)

        ### now reduce the columns to just nighttime hours
        mom_night = filter_outside_hours(all_mom, "DateTime", 7, 20)
        rfid_night = filter_outside_hours(all_rfid, "PIT_DateTime", 7, 20)

        if True:
            mom_night = format_time_cols(mom_night, date_fmt, cols=['DateTime'])
            rfid_night = format_time_cols(rfid_night, date_fmt, cols=['PIT_DateTime'])

        if True:
            populate_mom_Windows(mom_night[['DateTime', 'Burrow',  'Wt']]) 
            populate_RFID_Windows(rfid_night[['PIT_DateTime', 'Burrow', 'Rdr', 'PIT_ID']])

        # Display the joined DataFrame in t3
        join_widgetst1.delete('1.0', tk.END)  # Clear existing content
        join_widgetst1.insert(tk.END, joined_df.to_string(index=False))  # Insert joined DataFrame

        # Update the label showing the number of records in the joined DataFrame
        record_count = len(joined_df)
        # label_joined_records.config(text=f"Joined Records ({record_count})")


        ########
        # Saving the joined MOM and RFID DataFrame to a file
        #
        # Ask the user if they want to save the DataFrame
        if messagebox.askyesno("Save File", "Do you want to save the combined data to file?"):
            # Ask the user for file path and name
            output_file_path = filedialog.asksaveasfilename(
                title="Save Combined File As",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )

            if output_file_path:
                # Save df to file (tab-delimited, include headers)
                joined_df.to_csv(output_file_path, index=False, sep="\t")
                print(f"Data saved to {output_file_path}")
            else:
                print("Save cancelled.")
        else:
            print("User chose not to save the DataFrame.")



    except Exception as e:
        messagebox.showerror("Error", f"Failed to join MOM and RFID data: {e}")



###########################
#   function: join_MOM_RFID2
#       Joins MOM data to nearest RFID data within a specified time window.
#       - Uses a 2-minute window by default.
#       - Tries to find RFID data shifted by -1 hour first; if no matches, tries without shift.
#       - Keeps rows in hours <07:00 or >20:00 even if no matches found.
#       - Returns a DataFrame with columns sorted by Burrow and MOM_Time, including RFID, Rdr, and RFID_Time.
########
def join_MOM_RFID2(
    df_MOM: pd.DataFrame,
    df_RFID: pd.DataFrame,
    window: str = "2min",
    mom_time_col: str = "DateTime",
    rfid_time_col: str | None = None,
) -> pd.DataFrame:
    
    tol = pd.to_timedelta(window)

    # Pick RFID datetime column if not provided
    if rfid_time_col is None:
        if "PIT_DateTime" in df_RFID.columns:
            rfid_time_col = "PIT_DateTime"
        elif "DateTime" in df_RFID.columns:
            rfid_time_col = "DateTime"
        else:
            raise KeyError("No RFID datetime column found (need 'PIT_DateTime' or 'DateTime').")

    mom = df_MOM.copy()
    rfid = df_RFID.copy()

    # Ensure Burrow exists
    if "Burrow" not in mom.columns or "Burrow" not in rfid.columns:
        raise KeyError("Both df_MOM and df_RFID must contain a 'Burrow' column.")

    # Canonicalize burrow (digits only, last 3, zero-padded)
    def canon_burrow(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
             .str.replace(r"\D", "", regex=True)
             .str[-3:]
             .str.zfill(3)
        )

    mom["_BurrowKey"]  = canon_burrow(mom["Burrow"])
    rfid["_BurrowKey"] = canon_burrow(rfid["Burrow"])

    # Parse datetimes
    mom[mom_time_col]   = pd.to_datetime(mom[mom_time_col], errors="coerce")
    rfid[rfid_time_col] = pd.to_datetime(rfid[rfid_time_col], errors="coerce")

    # Build join times (shifted and zero)
    rfid["_join_time_shift"] = rfid[rfid_time_col] - pd.Timedelta(hours=1)
    rfid["_join_time_zero"]  = rfid[rfid_time_col]

    # Valid RFID per mode
    rfid_shift = rfid.dropna(subset=["_join_time_shift"])
    rfid_zero  = rfid.dropna(subset=["_join_time_zero"])

    # Count matches per MOM row (try shift, then zero)
    N = np.zeros(len(mom), dtype=int)
    used_offset = np.full(len(mom), -3600, dtype=int)  # -3600 (shift) by default; 0 for zero-shift

    tol_ns = np.int64(tol.value)
    mom_times_ns = mom[mom_time_col].to_numpy(dtype="datetime64[ns]")
    mom_bkeys    = mom["_BurrowKey"].to_numpy()

    # Precompute per-burrow arrays
    def arr_map(df: pd.DataFrame, col: str) -> dict:
        return {
            k: g[col].sort_values().to_numpy(dtype="datetime64[ns]")
            for k, g in df.groupby("_BurrowKey", sort=False)
        }

    arr_shift = arr_map(rfid_shift, "_join_time_shift")
    arr_zero  = arr_map(rfid_zero,  "_join_time_zero")

    def count_in(arr: np.ndarray, t_int: np.int64) -> int:
        if arr.size == 0:
            return 0
        low  = (t_int - tol_ns).view("datetime64[ns]")
        high = (t_int + tol_ns).view("datetime64[ns]")
        L = np.searchsorted(arr, low,  side="left")
        R = np.searchsorted(arr, high, side="right")
        return R - L

    for i, (t, bk) in enumerate(zip(mom_times_ns, mom_bkeys)):
        if np.isnat(t) or not isinstance(bk, (str, np.str_)):
            continue
        t_int = t.astype("int64")

        # Try shifted
        arr = arr_shift.get(bk)
        c = count_in(arr, t_int) if arr is not None else 0
        if c > 0:
            N[i] = c
            continue

        # Fallback: zero
        arr0 = arr_zero.get(bk)
        c0 = count_in(arr0, t_int) if arr0 is not None else 0
        if c0 > 0:
            N[i] = c0
            used_offset[i] = 0

    mom["n_Matches"] = N

    # Keep rows in target hours (even if N==0)
    valid_time = mom[mom_time_col].notna()
    hours_mask = valid_time & ((mom[mom_time_col].dt.hour < 7) | (mom[mom_time_col].dt.hour > 20))
    reduced = mom.loc[hours_mask].copy()

    # Attach per-row offset_used (aligned by index for safety)
    reduced["_offset_used"] = pd.Series(used_offset, index=mom.index).loc[reduced.index].values

    # --- Merge nearest per mode, per burrow key ---
    pieces = []
    for bk, mom_g in reduced.groupby("_BurrowKey", sort=False):

        # Split MOM rows by which offset they used (or default -3600)
        mom_shift = mom_g[mom_g["_offset_used"] == -3600].dropna(subset=[mom_time_col])
        mom_zero  = mom_g[mom_g["_offset_used"] == 0     ].dropna(subset=[mom_time_col])

        # Right sides for this burrow key
        r_shift_b = rfid_shift[rfid_shift["_BurrowKey"] == bk].sort_values("_join_time_shift", kind="mergesort")
        r_zero_b  = rfid_zero [rfid_zero ["_BurrowKey"] == bk].sort_values("_join_time_zero",  kind="mergesort")

        merged_parts = []

        if not mom_shift.empty and not r_shift_b.empty:
            m1 = pd.merge_asof(
                mom_shift.sort_values(mom_time_col, kind="mergesort"),
                r_shift_b[["_join_time_shift", "PIT_ID", "Rdr", "RF_File", "_BurrowKey"]],
                left_on=mom_time_col,
                right_on="_join_time_shift",
                by="_BurrowKey",
                direction="nearest",
                tolerance=tol,
            )
            m1["RFID_Time"] = m1["_join_time_shift"]
            merged_parts.append(m1)

        if not mom_zero.empty and not r_zero_b.empty:
            m2 = pd.merge_asof(
                mom_zero.sort_values(mom_time_col, kind="mergesort"),
                r_zero_b[["_join_time_zero", "PIT_ID", "Rdr", "RF_File", "_BurrowKey"]],
                left_on=mom_time_col,
                right_on="_join_time_zero",
                by="_BurrowKey",
                direction="nearest",
                tolerance=tol,
            )
            m2["RFID_Time"] = m2["_join_time_zero"]
            merged_parts.append(m2)

        if merged_parts:
            merged_g = pd.concat(merged_parts, ignore_index=False).sort_index(kind="mergesort")
        else:
            # No RFID rows for this burrow (or none within tolerance): keep MOM rows with NaNs
            merged_g = mom_g.copy()
            merged_g["PIT_ID"] = pd.NA
            merged_g["Rdr"] = pd.NA
            merged_g["RF_File"] = pd.NA
            merged_g["RFID_Time"] = pd.NaT

        merged_g["RFID"] = merged_g["PIT_ID"]
        pieces.append(merged_g)

    out = pd.concat(pieces, ignore_index=False).sort_index(kind="mergesort")

    # Rename to your schema
    rename_map = {}
    if "Trace_Segment_Num" in out.columns:
        rename_map["Trace_Segment_Num"] = "Segmnt"
    if "Wt_Min_Slope" in out.columns:
        rename_map["Wt_Min_Slope"] = "Wt"
    if "n_Matches" in out.columns:
        rename_map["n_Matches"] = "N"
    if "DateTime" in out.columns:
        rename_map["DateTime"] = "MOM_Time"
    out = out.rename(columns=rename_map)

    # Clean up helpers
    out = out.drop(columns=["_join_time_shift", "_join_time_zero", "PIT_ID"], errors="ignore")

    # Sort & order columns
    # out = out.sort_values(["Burrow", "MOM_Time"], kind="mergesort").reset_index(drop=True)

    desired_order = ["Burrow", "MOM_File", "MOM_Time", "Segmnt", "Wt", "RFID", "N", "Rdr", "RFID_Time", "RF_File"]
    out = out[[c for c in desired_order if c in out.columns]]



    out["Burrow"] = out["Burrow"].astype(str).apply(clean_burrow)

    out = format_time_cols(out, date_fmt, cols=['MOM_Time', 'RFID_Time'])


        # Sort by numeric Burrow when possible, then MOM_Time
    out["Burrow_sort"] = pd.to_numeric(out["Burrow"], errors="coerce")
    out = out.sort_values(["Burrow_sort", "MOM_Time"], kind="mergesort").drop(columns=["Burrow_sort"])

    return out


##########################
#   Setup GUI
#       All things tKinter related
#       Currently has:
#           One button to open a file
#           One popup menu to list days covered in the file
#           3 windows to display data
#           One menu (File) with one item (Quit)
########

def create_output_frame_var(parent, prefix: str = "", n: int = 3,
                        label_texts: list[str] | None = None,
                        width: int = 50, height: int = 50):
    """
    Build an output frame with N labeled columns, each with a scrollable Text.
    - prefix: key prefix for returned dict (e.g., 'mom_')
    - n: number of columns / Text widgets
    - label_texts: list of N strings for the header labels (trimmed/padded as needed)
    Returns: dict of widgets keyed like:
        {f"{prefix}frame", f"{prefix}label_1"... f"{prefix}t1", f"{prefix}t1_scrollbar", ...}
    """

    # Prepare labels
    if label_texts is None:
        label_texts = ["All Records", "One day", "Bird Activity"] if n == 3 \
                      else [f"Section {i+1}" for i in range(n)]
    else:
        label_texts = list(label_texts[:n]) + [f"Section {i+1}" for i in range(len(label_texts), n)]

    frame = tk.Frame(parent, width=width, height=height, bd=1, relief=tk.SOLID)
    frame.pack_propagate(False)

    labels: dict[str, tk.Label] = {}
    text_widgets: dict[str, tk.Text] = {}
    scrollbars: dict[str, tk.Scrollbar] = {}

    # Header labels + columns
    for col in range(n):
        # Header
        lbl_key = f"label_{col+1}"
        labels[lbl_key] = tk.Label(frame, text=label_texts[col], font=("Arial", 12))
        labels[lbl_key].grid(row=0, column=col, padx=10, pady=(10, 5))

        # Column container
        sub = tk.Frame(frame, bd=1, relief=tk.SOLID)
        sub.grid(row=1, column=col, padx=10, pady=5, sticky=tk.NSEW)

        # Scrollbar + Text
        sb = Scrollbar(sub, orient="vertical")
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        t_key = f"t{col+1}"
        txt = tk.Text(sub, width=width, height=height, yscrollcommand=sb.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=txt.yview)

        text_widgets[t_key] = txt
        scrollbars[f"{t_key}_scrollbar"] = sb

        # Make each column expand evenly
        frame.grid_columnconfigure(col, weight=1)

    # Let the text row expand vertically
    frame.grid_rowconfigure(1, weight=1)

    return {
        f"{prefix}frame": frame,
        **{f"{prefix}{k}": v for k, v in labels.items()},
        **{f"{prefix}{k}": v for k, v in text_widgets.items()},
        **{f"{prefix}{k}": v for k, v in scrollbars.items()},
    }


##########################
#   function: create_output_frame
#    Create an output frame with 3 labeled sections and scrollable text boxes.
#    prefix: optional string to prefix variable names (e.g., 'mom_')
#    Returns a dictionary of widget references.
########
def create_output_frame(parent, prefix="", width=50, height=50):

    frame = tk.Frame(parent, width=width, height=height, bd=1, relief=tk.SOLID)
    frame.pack_propagate(False)  # Keep specified size

    # Labels
    labels = {}
    labels["label_all_records"] = tk.Label(frame, text="All Records", font=("Arial", 12))
    labels["label_all_records"].grid(row=0, column=0, padx=10, pady=(10, 5))

    labels["one_day"] = tk.Label(frame, text="One day", font=("Arial", 12))
    labels["one_day"].grid(row=0, column=1, padx=10, pady=(10, 5))

    labels["bird_activity"] = tk.Label(frame, text="Bird Activity", font=("Arial", 12))
    labels["bird_activity"].grid(row=0, column=2, padx=10, pady=(10, 5))

    # Text areas
    text_widgets = {}
    scrollbars = {}
    for i, key in enumerate(["t1", "t2", "t3"], start=0):
        sub_frame = tk.Frame(frame, bd=1, relief=tk.SOLID)
        sub_frame.grid(row=1, column=i, padx=10, pady=5, sticky=tk.NSEW)

        scrollbar = Scrollbar(sub_frame, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(sub_frame, width=width, height=height, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widgets[key] = text_widget
        scrollbars[f"{key}_scrollbar"] = scrollbar

    return {
        f"{prefix}frame": frame,
        **{f"{prefix}{k}": v for k, v in labels.items()},
        **{f"{prefix}{k}": v for k, v in text_widgets.items()},
        **{f"{prefix}{k}": v for k, v in scrollbars.items()},
    }

##########################
#   function: assign_widget_refs
#       Assign each widget in widget_dict to a variable with the same name as the key
#       Defaults to the caller's global namespace
########
def assign_widget_refs(widget_dict, namespace=None):

    import inspect
    
    if namespace is None:
        # Get the caller's global namespace
        namespace = inspect.currentframe().f_back.f_globals
    
    for name, widget in widget_dict.items():
        namespace[name] = widget
        # Optional: print for verification
        print(f"Assigned variable: {name} -> {widget}")




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
window_height = int(screen_height * 0.8)

root.geometry(f"{window_width}x{window_height}")

##########################
# Create a parent container for RFID + MOM buttons side-by-side
##########################
buttons_side_by_side = tk.Frame(root)
buttons_side_by_side.pack(side=tk.TOP, pady=(20, 0), fill=tk.X)

# Configure two equal-width columns so both containers match size
buttons_side_by_side.grid_columnconfigure(0, weight=1, uniform="buttons")
buttons_side_by_side.grid_columnconfigure(1, weight=1, uniform="buttons")

##########################
# RFID Button Container
##########################
buttons_rfid_container = tk.Frame(
    buttons_side_by_side,
    bd=0,
    relief=tk.SOLID,
)
buttons_rfid_container.grid(row=0, column=0, padx=(0, 25), sticky="nsew")  # gap to MOM side

frame_buttons_date = tk.Frame(buttons_rfid_container)
frame_buttons_date.pack(side=tk.TOP, fill=tk.X, padx=20, pady=20)

# Buttons for RFID
if False:
    button_load_file = tk.Button(frame_buttons_date, text="Load File", command=load_file)
    button_load_file.pack(side=tk.RIGHT)

button_load_all_RFID = tk.Button(frame_buttons_date, text="Load RFID Files", command=load_all_RFID_files)
button_load_all_RFID.pack(side=tk.RIGHT)



if False:
    button_load_file = tk.Button(frame_buttons_date, text="Birds", command=update_Unique_Tags_menu)
    button_load_file.pack(side=tk.RIGHT)

    mb_days = tk.Menubutton(frame_buttons_date, text="Days", indicatoron=True, borderwidth=1, relief="raised")
    mb_days.pack(side=tk.RIGHT, padx=10)
    days_menu = tk.Menu(mb_days, tearoff=0)
    mb_days.configure(menu=days_menu)

    mb_pit = tk.Menubutton(frame_buttons_date, text="RFID", indicatoron=True, borderwidth=1, relief="raised")
    mb_pit.pack(side=tk.RIGHT, padx=10)
    pit_menu = tk.Menu(mb_pit, tearoff=0)
    mb_pit.configure(menu=pit_menu)

##########################
# MOM Button Container
##########################
buttons_mom_container = tk.Frame(
    buttons_side_by_side,
    bd=0
)
buttons_mom_container.grid(row=0, column=1, padx=(25, 0), sticky="nsew")  # gap to RFID side

mom_frame_buttons = tk.Frame(buttons_mom_container)
mom_frame_buttons.pack(side=tk.TOP, fill=tk.X, padx=20, pady=20)

# Buttons for MOM
mom_button_load_all = tk.Button(mom_frame_buttons, text="Load MOM Files", command=load_all_MOM_files)
mom_button_load_all.pack(side=tk.LEFT)

mom_button_load_file = tk.Button(mom_frame_buttons, text="Join RFID+MOM Data", command=do_Join_MOM_RFID   ) # make this one file only. debug
mom_button_load_file.pack(side=tk.LEFT)



if False:
    mom_button_birds = tk.Button(mom_frame_buttons, text="Birds", command=update_Unique_Tags_menu)
    mom_button_birds.pack(side=tk.LEFT)

    mom_mb_days = tk.Menubutton(mom_frame_buttons, text="Days", indicatoron=True, borderwidth=1, relief="raised")
    mom_mb_days.pack(side=tk.LEFT, padx=10)
    mom_days_menu = tk.Menu(mom_mb_days, tearoff=0)
    mom_mb_days.configure(menu=mom_days_menu)

    mom_mb_pit = tk.Menubutton(mom_frame_buttons, text="MOM RFID", indicatoron=True, borderwidth=1, relief="raised")
    mom_mb_pit.pack(side=tk.LEFT, padx=10)
    mom_pit_menu = tk.Menu(mom_mb_pit, tearoff=0)
    mom_mb_pit.configure(menu=mom_pit_menu)


##########################
#   Create the output frames 
# ####
output_container = tk.Frame(root)
output_container.pack(side=tk.TOP, pady=(20, 0))  # 20 points below button frames

if False:
    pass
else:
    # Three columns with your original labels
    output_widgets = create_output_frame_var(output_container, n=1, height=20,
                                        label_texts=["RFIDs"])
    output_widgets["frame"].grid(row=0, column=0, padx=(0, 25), pady=0) 

    # Two columns for MOM
    mom_widgets = create_output_frame_var(output_container, prefix="mom_", n=1,height=20,
                                    label_texts=["Traces"])
    mom_widgets["mom_frame"].grid(row=0, column=1, padx=(25, 0), pady=0)

    # Make a join container for the joined data
    # Create the join_container frame (1 text widget, width=150, custom label)
    join_widgets = create_output_frame_var(
        parent=root,                      # or the parent container you want it under
        prefix="join_widgets",                   # prefix for widget keys,
        n=1,                              # only 1 text widget
        width=140,                        # wider text widget
        height=50,                        # same height as others
        label_texts=["MOM Traces / RFID"]       # custom label
)

# Place join_container below the previous container
# join_widgets["join_widgets_frame"].pack(side=tk.TOP, pady=(20, 0))
join_widgets["join_widgetsframe"].pack(side=tk.TOP, pady=(20, 0))


# Make them a font I can see
set_text_widget_font(output_widgets, font_name="Courier", font_size=16)
set_text_widget_font(mom_widgets, font_name="Courier", font_size=16)
set_text_widget_font(join_widgets, font_name="Courier", font_size=16)


# Assign references into the global namespace
assign_widget_refs(output_widgets)
assign_widget_refs(mom_widgets)
assign_widget_refs(join_widgets)
print(join_widgets)

# Keep output frames centered
output_container.grid_columnconfigure(0, weight=1)
output_container.grid_columnconfigure(1, weight=1)



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
