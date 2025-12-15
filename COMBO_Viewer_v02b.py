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

# RFID tag ID length for use in validation of incoming data
gExpected_PIT_ID_Len = 10

# time tolerance for matching MOM to RFID in minutes - pd.to_timedelta will accept a number, but it interprets a bare number as nanoseconds unless you also pass unit
gTimeTolerance = "70min"

# variables for time periods when we are 100% sure birds show up at night - 
gNightStartHour = 19  # 8 PM
gNightEndHour = 7     # 6 AM

global do_print
do_print = False  # Set to True to enable print statements for debugging
Show_Buttons = False  # Toggle MOM button visibility to give more space to outputs
myTesting = False
# vTesting_Folder = "/Users/bobmauck/devel/Combo_App/Example_Data"  # folder for testing on Mauck computer
vTesting_Folder = "/Users/bobmauck/Dropbox/BIG_Science/MOMs/Testing/Sam_Data"  # folder for testing from Sam's Google Drive data

global vVersString
global vAppName
vVersString = " (v_02.1b)"  ## upDATE AS NEEDED - v01 Beta for testing
vAppName = "Combo Viewer" + vVersString
if do_print:
    print(f"Starting {vVersString} - {vAppName}")
########   Versions
#       v_01.0 - initial working version
#       v_01.1 - minor change: fixed bug that caused error on non-Mauck computers - file paths
#       v_01.2b - added menus and setup.py for building Mac app with py2app
#       v_02.0b - Cleaned up code, UI changes
#       v_02.1b - Added functions to changing the window for finding rFid, other changes
################

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

##########################
#   remove_spurious_pairs
#       receives dataframe with [MOM] data and the column of interest df_MOM['Wt' or 'Wt_Min_Slope']
#       finds contiguous measures that are in the pattern low_val, high_val which are calibrations
#       removes them from the df and returns the clean df
########
def remove_spurious_pairs(
    df: pd.DataFrame,
    col: str = "Wt_Min_Slope",
    low_val: float = 50.0,
    high_val: float = 95.0,
    tol: float = 0.6
) -> pd.DataFrame:

    df = df.copy()

    # Step 1: detect low_val followed by high_val+
    near_low = df[col].between(low_val - tol, low_val + tol)
    over_high = df[col] > high_val

    # pattern[i] = True if row i ≈ low_val and row i+1 > high_val
    pattern = near_low & over_high.shift(-1)

    # rows to drop: both the ≈low_val row and its following >high_val row
    drop_idx = df.index[pattern | pattern.shift(1, fill_value=False)]

    # Step 2: also remove any other stray > high_val rows
    drop_idx = drop_idx.union(df.index[df[col] > high_val])

    return df.drop(drop_idx).reset_index(drop=True)


#############
#   return_useful_name: takes a path string and returns just the name of the file
#     used for displaying the file name in the GUI
####
def return_useful_name(the_path):
    where = the_path.rfind("/")
    the_name = the_path[(where + 1):(len(the_path)-4)]
    return the_name

#############
# clean_burrow: takes the last 7 characters of a filename and returns the burrow number
#     - removes file extension if present (.txt, .csv, any case)    
#     - takes the last 3 characters
#     - returns the burrow number padded to 3 digits becuase we treat it as charccters
####
def clean_burrow(val: str) -> str:
    s = str(val)

    # Remove file extension if present (.txt, .csv, any case)
    for ext in (".txt", ".csv"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    # take the last 3 of what is left
    s = s[-3:]

    # deal with less than 3 digits
    if s.startswith("_"):       # e.g., "_31"
        s = s[1:]               # drop the leading underscore → "31"
    elif "_" in s:              # e.g., "5_3"
        s = s.split("_")[-1]    # take part after underscore → "3"

    # Return only if it's digits and padded to 3 places
    return s.zfill(3)


###########################
#   function: clear_text_widgets
#       Clears the content of all Text widgets in the given widget dictionary
#       utility function for GUI
#########
def clear_text_widgets(widget_dict):
    for key, widget in widget_dict.items():
        if isinstance(widget, tk.Text) and key.lstrip("_").split("_")[-1].startswith("t"):
            widget.delete("1.0", tk.END)

###########################
#   function: set_text_widget
#       Sets the font for all Text widgets in the given widget dictionary
#       Sets the font to the specified font name and size
#       Parameters:
#           - output_frame_dict (dict): Dictionary of widgets returned from create_output_frame()
#           - font_name (str): Font family name (default: Arial)
#           - font_size (int): Font size in points (default: 14)
#       utility function for GUI
#########
def set_text_widget_font(output_frame_dict, font_name="Arial", font_size=14):
    for widget in output_frame_dict.values():
        if isinstance(widget, tk.Text):
            widget.config(font=(font_name, font_size))


##########################
#   function: populate_RFID_Windows
#       Takes a dataframe and puts it in the t1 window
#       Also updates the Days menu and Unique Tags menu
#       Updates the label showing the number of records
#       utility function for GUI
########
def populate_RFID_Windows(df_rfid):

    if df_rfid.empty:
        messagebox.showwarning("Warning", "DataFrame is empty. Please load data first.")
        return
    
    # Sort by DateTime
    df_rfid = df_rfid.sort_values(by=["Burrow", "PIT_DateTime"]).reset_index(drop=True)

    # Clear existing content in t1, t2, and t3
    clear_text_widgets(output_widgets)
    
    # Insert the entire dataframe into t1
    # t1.insert(tk.END, df_rfid.to_string(index=False))
    table_str = format_df_custom(df_rfid, mode="RFID")
    t1.insert(tk.END, table_str)

    # Update the label showing the number of records
    try:
        record_count = len(df_rfid)
        label_1.config(text=f"RFIDs ({record_count} records)")
    except Exception:
        pass

def format_df_custom(df, mode="MOM"):
    """
    Format a DataFrame for display in Tkinter Text widget.
    
    mode = "MOM"  → expects columns: Burrow, DateTime, Wt
    mode = "RFID" → expects columns: Burrow, PIT_DateTime, Rdr, PIT_ID
    mode = "JOIN" → expects columns:
        Burrow, MOM_File, MOM_Time, Wt, RFID, N, Rdr,
        Closest_RFID_Time, RF_File
    """

    if mode == "MOM":
        widths = {"Burrow": 10, "DateTime": 25, "Wt": 10}
        aligns = {"Burrow": "^", "DateTime": "^", "Wt": ">"}
        cols = ["Burrow", "DateTime", "Wt"]

    elif mode == "RFID":
        widths = {"Burrow": 8, "PIT_DateTime": 20, "Rdr": 6, "PIT_ID": 12}
        aligns = {"Burrow": "^", "PIT_DateTime": "^", "Rdr": "^", "PIT_ID": ">"}
        cols = ["Burrow", "PIT_DateTime", "Rdr", "PIT_ID"]

    elif mode == "JOIN":
        widths = {
            "Burrow": 6,
            "MOM_File": 25,
            "MOM_Time": 20,
            "Wt": 8,
            "RFID": 12,
            "N": 4,
            "Rdr": 5,
            "Closest_RFID_Time": 20,
            "RF_File": 25
        }
        aligns = {
            "Burrow": "^",
            "MOM_File": "<",
            "MOM_Time": "^",
            "Wt": ">",
            "RFID": ">",
            "N": ">",
            "Rdr": "^",
            "Closest_RFID_Time": "^",
            "RF_File": "<"
        }
        cols = [
            "Burrow", "MOM_File", "MOM_Time", "Wt",
            "RFID", "N", "Rdr", "Closest_RFID_Time", "RF_File"
        ]

    else:
        raise ValueError("mode must be 'MOM', 'RFID', or 'JOIN'")

    # Ensure only desired columns
    df = df[cols].copy()

    # Build header
    header = " ".join(
        f"{col:{aligns.get(col, '<')}{widths.get(col, 12)}}" for col in cols
    )
    lines = [header, "-" * len(header)]

    if False:
        # Build rows
        for _, row in df.iterrows():
            line = " ".join(
                f"{str(row[col]):{aligns.get(col, '<')}{widths.get(col, 12)}}" for col in cols
            )
            lines.append(line)
    else:
        for _, row in df.iterrows():
            formatted_values = []
            for col in cols:
                val = row[col]

                # --- Special formatting ---
                if col == "Wt" and pd.notna(val):
                    try:
                        val = f"{float(val):.2f}"
                    except Exception:
                        val = str(val)

                elif "Time" in col and pd.notna(val):  # handles DateTime, MOM_Time, PIT_DateTime, Closest_RFID_Time
                    try:
                        val = pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        val = str(val)

                else:
                    val = str(val) if pd.notna(val) else ""

                # Apply alignment + width
                formatted_values.append(
                    f"{val:{aligns.get(col, '<')}{widths.get(col, 12)}}"
                )

            lines.append(" ".join(formatted_values))


    return "\n".join(lines)



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

    if False:
        # Insert into t1, but space and justify
        df_mom.to_string(index=False, col_space={"Burrow": 5, "DateTime": 25, "Wt": 20}, justify="right")
        mom_t1.insert(tk.END, df_mom.to_string(index=False))
        # table_str = df_mom.to_string(index=False, col_space=15, justify="center")
        #mom_t1.insert(tk.END, table_str)
    else:
        # Insert the entire dataframe into t1
         # Format dataframe with custom spacing/justification
        table_str = format_df_custom(df_mom, "MOM")
        mom_t1.insert(tk.END, table_str)

    # Update the label showing the number of records
    try:
        record_count = len(df_mom)
        mom_label_1.config(text=f"Traces ({record_count} records)")
    except Exception:
        pass

##########################
#   function: load_all_MOM_files
#       GUI Button Call: Calls get_All_MOM_data to get all MOM data in a folder
#       Puts it in the global dataframe
#       Then populates the windows with that data
########    
def load_all_MOM_files():
    global all_mom # use this to hold the dataframe for all MOM data for combo work?
    df_all_mom = get_All_Mom_data()
    # remove known calibratoin data which occur in pairs 

    if df_all_mom.empty:
        print("No matching files found.")
        return pd.DataFrame(columns=['MOM_File', 'Segment', 'DateTime', 'Wt', 'Burrow'])

    # Combine and drop duplicates
    df_all_mom = df_all_mom.drop_duplicates().reset_index(drop=True)

    df_MOM = df_all_mom.copy()



    # Rename columns for consistency
    df_MOM.rename(columns={
        'File': 'MOM_File',
        # 'Trace_Segment_Num': 'Segment',
        'Wt_Min_Slope': 'Wt'
    }, inplace=True)

    df_valid_mom = remove_spurious_pairs(df_MOM, col="Wt", low_val=50, high_val=80, tol=1.0)

        # Sort by MOM_File then MOM_Time
    df_valid_mom = df_valid_mom.sort_values(
        by=["Burrow", "MOM_File", "DateTime"], ignore_index=True
    )

    # Make sure DateTime is coerced to datetime (bad values become NaT)
    df_valid_mom["DateTime"] = pd.to_datetime(df_valid_mom["DateTime"], errors="coerce")

    # Drop rows where DateTime is NaT (invalid or gibberish)
    df_valid_mom = df_valid_mom.dropna(subset=["DateTime"])


    # print(df_all_rfid.head(10))  # Print first 10 rows for verification
    # populate_mom_Windows(df_all_mom[['MOM_File', 'Segment', 'DateTime', 'Wt_Min_Slope', 'Burrow']]) # only if we have wider text window
    print("populate_mom_Windows")
    populate_mom_Windows(df_valid_mom[['Burrow', 'DateTime',  'Wt']]) # moved this code to this function 7/18/2024 - can use it with one file or many files

    #### get a list of unique burrows
    burrow_df = pd.DataFrame({"Burrow": sorted(df_valid_mom["Burrow"].dropna().unique())})
    
    # return df_valid_mom, burrow_df
    return df_valid_mom 


##########################
#   function: get_All_RFID_data - from Jup Notebook 1:40PM on 8/23/25
#       Mac interface to open all RFID files in a folder - once used this folder: str = "/Users/bobmauck/devel/Combo_App/Example_Data"
#       Once opened, put in dataframe and display
#           Load all files in 'folder' that start with 'RF' and end with '.txt',
#       combine them into a single DataFrame, remove duplicates, and return it.
#       Assumes the files have NO header row and uses:
#           ['PIT_ID', 'Rdr', 'PIT_DateTime'] as the column names.
########
def get_All_RFID_data(folder: str = None, one_Burr: str = None) -> pd.DataFrame:
    """
    Load all files in 'folder' that start with 'RF' and end with '.txt',
    combine them into a single DataFrame, remove duplicates, and return it.
    Assumes the files have NO header row and uses:
        ['PIT_ID', 'Rdr', 'PIT_DateTime']
    as the column names.
    """

    print("here is the folder chosen for RFID data:", folder)

    if folder is not None:
        folder = vTesting_Folder
    else:
        messagebox.showwarning("Find Data", "Choose folder with RFID data.")
        folder = filedialog.askdirectory()
        if not folder:  # User cancelled the dialog
            messagebox.showwarning("Find Data", "No Folder Selected")
            return None

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
                OUT_FMT = "%m/%d/%Y %H:%M:%S" #output format for PIT_DateTime regardless of incoming format
                EXPECTED_LEN = gExpected_PIT_ID_Len  # Lenght of a PIT_ID - make sure this doesn't change - or have a pref?

                df_temp = pd.read_csv(
                    file_path,
                    delimiter=',',
                    header=None,                # No header in file
                    names=cols_to_import,       # Assign column names
                    usecols=[0, 1, 2],          # Only first 3 columns
                    on_bad_lines='warn',
                    encoding="latin1",          # or "utf-16" if that file is UTF-16
                    encoding_errors="replace",  # keep rows, substitute bad bytes

                )

                # Add the filename column
                df_temp['RF_File'] = filename
                df_temp['Burrow'] = df_temp['RF_File'].astype(str).str[-7:-4]
                df_temp["Burrow"] = df_temp["Burrow"].astype(str).apply(clean_burrow)

                # Normalize types for string operations
                df_temp["PIT_ID"] = df_temp["PIT_ID"].astype(str)
                df_temp["Rdr"] = df_temp["Rdr"].astype(str)
                df_temp["PIT_DateTime"] = df_temp["PIT_DateTime"].astype(str)

                # Format PIT_DateTime to standard format
                df_temp["PIT_DateTime"] = pd.to_datetime(
                    df_temp["PIT_DateTime"],
                    errors="coerce",  # handles both "06/14/2025 16:46:52" and "2025-07-26T14:00:00"
                    format="mixed"
                )
                df_temp["PIT_DateTime"] = df_temp["PIT_DateTime"].dt.strftime(OUT_FMT).fillna("")

                # remove rows with invalid PIT_ID length, etc.
                df_temp = df_temp[
                    ~df_temp["PIT_ID"].str.upper().isin({"STARTUP", "RUNNING"})
                    & pd.to_numeric(df_temp["Rdr"], errors="coerce").notna()
                    & df_temp["PIT_ID"].str.len().eq(EXPECTED_LEN)
                ].copy()
                # reset index after filtering
                df_temp = df_temp.reset_index(drop=True)
                
                all_dfs.append(df_temp)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_dfs:
        print("No matching files found.")
        return pd.DataFrame(columns=cols_to_import)

    # Combine and drop duplicates
    df_RFID = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    if one_Burr is not None:
        target = str(one_Burr).zfill(3)
        df_RFID = df_RFID[df_RFID["Burrow"] == target].copy()

    return df_RFID

##########################
#   function: load_all_RFID_files
#       GUI Call button: Calls get_All_RFID_data to get all RFID data in a folder
#       Puts it in the global dataframe
#       Then populates the windows with that data
########    
def load_all_RFID_files():
    # global all_rfid # use this to hold the dataframe for all RFID data for combo work?
    df_all_rfid = get_All_RFID_data()
    
    if df_all_rfid.empty:
        messagebox("No RFID folder chosen")
        return None
    else:
        df_all_rfid = format_time_cols(df_all_rfid, date_fmt, cols=['PIT_DateTime'])
        # chg   
        # df_all_rfid["Burrow"] = df_all_rfid["Burrow"].astype(str).apply(clean_burrow)

        # Ensure PIT_DateTime is actually datetime
        df_all_rfid["PIT_DateTime"] = pd.to_datetime(df_all_rfid["PIT_DateTime"], errors="coerce")

        # Drop rows where PIT_DateTime is missing (NaT)
        df_all_rfid = df_all_rfid.dropna(subset=["PIT_DateTime"])
        # Drop rows where burrow is not right 
        df_all_rfid = df_all_rfid[df_all_rfid["Burrow"].astype(str).str.isdigit()].copy()

        # Sort by RFID_File then Time
        df_all_rfid = df_all_rfid.sort_values(
            by=["Burrow", "PIT_DateTime"], ignore_index=True
        )

        populate_RFID_Windows(df_all_rfid[['PIT_DateTime', 'Burrow', 'Rdr', 'PIT_ID']])
        return df_all_rfid 

##########################
#   function: earliest_per_pit
"""
    Return a subset of df containing the earliest PIT_DateTime
    record for each unique PIT_ID.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['PIT_ID', 'PIT_DateTime'].

    Returns
    -------
    pd.DataFrame
        Subset with earliest record for each PIT_ID.
    """
######## 
def earliest_per_pit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop the all-zero PIT_ID rows if doing so leaves data
    filtered = df[df["PIT_ID"] != "0000000000"].copy()
    if not filtered.empty:
        df = filtered

    # make sure PIT_DateTime is datetime
    df["PIT_DateTime"] = pd.to_datetime(df["PIT_DateTime"], errors="coerce")

    # sort by PIT_ID then datetime
    df = df.sort_values(["PIT_ID", "PIT_DateTime"])

    # keep first row for each PIT_ID
    earliest = df.groupby("PIT_ID", as_index=False).first()

    return earliest.reset_index(drop=True)


##########################
#   function: resolve_missing_RFIDs
########
def resolve_missing_RFIDs(df_Missing_MOMs: pd.DataFrame,
                          df_rfids: pd.DataFrame) -> pd.DataFrame:
    # if False: print("inside resolve_missing")

    # accumulator list
    resolved_chunks = []

    # ensure PIT_DateTime is datetime in RFID file
    df_all_rfids = df_rfids.copy()
    df_all_rfids['PIT_DateTime'] = pd.to_datetime(df_all_rfids['PIT_DateTime'], errors="coerce")

    # unique MOM file names
    unique_files = df_Missing_MOMs['MOM_File'].unique()
    # if False:
    #     print("unique files:")
    #     print(unique_files[:10])

    for mom_file in unique_files:
        # subset mom records for this file
        mom_temp = df_Missing_MOMs[df_Missing_MOMs['MOM_File'] == mom_file].copy()
        if mom_temp.empty:
            continue

        # get focal date string from MOM_File (e.g., "DL_06_25_2025_87.TXT")
        my_Day_str = mom_file[3:13]
        focal_date = pd.to_datetime(my_Day_str, format="%m_%d_%Y", errors="coerce")

        # time window: 20:00 same day → 07:00 next day - make a pref for this
        start = focal_date + pd.Timedelta(hours=gNightStartHour)
        end   = focal_date + pd.Timedelta(days=1, hours=gNightEndHour)
        # if False:
        #     print("start and end time:")
        #     print(start)
        #     print(end)

        # Burrow for these MOM records
        my_burr = mom_temp['Burrow'].iloc[0]
        # if False: print(f"N RFIDs: {len(df_rfids)}")

        # candidate RFID records
        rfid_temp = df_all_rfids[
            (df_all_rfids['Burrow'] == my_burr) &
            (df_all_rfids['PIT_DateTime'] >= start) &
            (df_all_rfids['PIT_DateTime'] <= end)
        ].copy()

        # add column N = number of candidate RFID records
        rfid_temp["N"] = len(rfid_temp)

        # keep earliest PIT record per PIT_ID
        rfid_temp = earliest_per_pit(rfid_temp)

        if rfid_temp.empty:
            # Add placeholder RFID columns — but MOM_Time stays as in mom_temp
            mom_temp = mom_temp.assign(
                RFID=pd.NA,
                N=0,  # explicitly mark no RFID found
                Rdr=pd.NA,
                Closest_RFID_Time=pd.NaT,
                RF_File=pd.NA
            )
            # if False: print(f"no RFID found for {my_Day_str}_{my_burr}")

        else:
            # Select and rename RFID columns to match desired schema
            rfid_temp = rfid_temp.rename(columns={
                "PIT_ID": "RFID",
                "PIT_DateTime": "Closest_RFID_Time",
                "RF_File": "RF_File",
                "Rdr": "Rdr"
            })[["RFID", "N", "Rdr", "Closest_RFID_Time", "RF_File"]]

            if len(mom_temp) == len(rfid_temp):
                mom_temp = pd.concat([mom_temp.reset_index(drop=True),
                                      rfid_temp.reset_index(drop=True)], axis=1)
                # if False:
                #     print("printing 1:1 ratio outcome:")
                #     print(mom_temp)

            elif len(mom_temp) > 1 and len(rfid_temp) == 1:
                # Repeat RFID row to match length of mom_temp
                rfid_repeated = pd.concat([rfid_temp] * len(mom_temp), ignore_index=True)
                mom_temp = pd.concat([mom_temp.reset_index(drop=True),
                                      rfid_repeated.reset_index(drop=True)], axis=1)
                # if False:
                #     print("printing 1:many outcome:")
                #     print(mom_temp)

            else:
                # fallback: MOM row only, empty RFID fields
                mom_temp = mom_temp.assign(
                    RFID=pd.NA,
                    N=len(rfid_temp),
                    Rdr=pd.NA,
                    Closest_RFID_Time=pd.NaT,
                    RF_File=pd.NA
                )

        # 🔑 MOM_Time is never touched or recalculated — original value is preserved
        resolved_chunks.append(mom_temp)

    # collapse all results into a single DataFrame
    if resolved_chunks:
        df_resolved_moms = pd.concat(resolved_chunks, ignore_index=True)
    else:
        df_resolved_moms = pd.DataFrame(columns=[
            "Burrow", "MOM_File", "MOM_Time", "Segment", "Wt_Min_Slope",
            "RFID", "N", "Rdr", "Closest_RFID_Time", "RF_File"
        ])

    # 🔑 Rename Wt_Min_Slope → Wt
    if "Wt_Min_Slope" in df_resolved_moms.columns:
        df_resolved_moms = df_resolved_moms.rename(columns={"Wt_Min_Slope": "Wt"})

    # enforce final schema & column order
    desired_cols = ["Burrow", "MOM_File", "MOM_Time", "Segment", "Wt",
                    "RFID", "N", "Rdr", "Closest_RFID_Time", "RF_File"]
    df_resolved_moms = df_resolved_moms.reindex(columns=desired_cols)

    # if False:
    #     print("df_resolved_moms total")
    #     print(df_resolved_moms.head(3))

    return df_resolved_moms



def build_final_combo_MOM_RFID(df_WtFiles: pd.DataFrame,
                               df_rfid: pd.DataFrame) -> pd.DataFrame:
    """
    Build final combined MOM + RFID dataframe:
      1. Run join_MOM_RFID2 to get df_MOM_with_counts
      2. Find missing-time MOM rows, filter with remove_spurious_pairs,
         and resolve them with resolve_missing_RFIDs
      3. Append resolved rows back to df_MOM_with_counts
      4. Sort final DataFrame by MOM_File then MOM_Time
      5. Enforce final schema/column order
      6. Add Matched flag (True if RFID match found, False otherwise)
    """

    desired_cols = [
        "Burrow", "MOM_File", "MOM_Time", "Wt",
        "RFID", "N", "Rdr", "Closest_RFID_Time", "RF_File", "Matched"
    ]

    in_debug = True
    if in_debug:
        print(f"Number of rows before JoinRFID2: {len(df_WtFiles)}")

    # 1. Initial join
    # df_MOM_with_counts = join_MOM_RFID2(df_WtFiles, df_rfid, window=gTimeTolerance)
    df_MOM_with_counts = join_MOM_RFID_New(df_WtFiles, df_rfid, window=gTimeTolerance)
    df_MOM_with_counts["Matched"] = True

    if in_debug:
        print(f"Number of rows after JoinRFID2: {len(df_MOM_with_counts)}")

    # --- 🔑 Standardize naming ---
    df_MOM_with_counts = df_MOM_with_counts.rename(
        columns={
            "Wt_Min_Slope": "Wt",
            "RFID_Time": "Closest_RFID_Time",
            "Segmnt": "Segment"
        }
    )

    # 2. Identify missing-time rows
    df_WtFiles = df_WtFiles.copy()
    df_WtFiles["DateTime"] = pd.to_datetime(df_WtFiles["DateTime"], errors="coerce")

    if False:
        n_records_87Before = len(df_WtFiles[df_WtFiles["Burrow"] == "087"])
        print("Number of records with Burrow == '087':", n_records_87Before)


    df_missing_time = df_WtFiles[
        df_WtFiles["DateTime"] == pd.Timestamp("2000-01-01 00:00:00")].copy()
        ## also catch NaT or Windows time null?

    #### --- CHANGE -- remove df_missing_time from df_WtFiles
    df_WtFiles = df_WtFiles[df_WtFiles["DateTime"] != pd.Timestamp("2000-01-01 00:00:00")].copy()


    if False:
        n_records_87after = len(df_missing_time[df_missing_time["Burrow"] == "087"])
        print("Number of records '087' with missing time:", n_records_87after)

    if False: 
        print("df_missing_time #recs ", len(df_missing_time))
        # print(df_missing_time.head(10))

    valid_missing = remove_spurious_pairs(
        df_missing_time, "Wt_Min_Slope", low_val=50, high_val=75, tol=0.6
    )

    if False:
        n_records_87after = len(df_missing_time[df_missing_time["Burrow"] == "087"])
        print("Number of records '087' after removing spurious:", n_records_87after)

    # 3. Resolve missing rows
    df_r = resolve_missing_RFIDs(valid_missing, df_rfid) #----- this is Correct - right number/and linked

    if in_debug:
        print(f"Number of rows after remove_spurious: {len(df_r)}")

    if False:
        n_records_87after = len(df_r[df_r["Burrow"] == "087"])
        print("Number of records '087' after removing resolve_missing_RFIDS:", n_records_87after)
        records_087 = df_r[df_r["Burrow"] == "087"]
        print(records_087)
        print("--------")


    if False: 
        print("df_rhead")
        print(df_r.head(5))
    
    if not df_r.empty:
        if False: print ("df_r is NOT empty") 
        df_r = df_r.rename(
            columns={"Wt_Min_Slope": "Wt", "PIT_Time": "Closest_RFID_Time"}
        )
        df_r["Matched"] = df_r["RFID"].notna()
    else:
        if False: print("df_r is full")
        df_r = pd.DataFrame(columns=desired_cols)

    # losing something here lost df_r values 

    # --- 🔑 Ensure both dfs have same schema ---
    df_MOM_with_counts = df_MOM_with_counts.reindex(columns=desired_cols)
    ## remove all bad dates becaues they are restored with df_r


    df_r = df_r.reindex(columns=desired_cols)

    if False:
        print("--- head for MOMs and RFID before concat")
        print(df_MOM_with_counts.head(3))
        print(df_r.head(3))
    # 4. Combine - this is the problem. we want to keep the empty MOM RTCs, but they are overriding the resolved. Resolved should have them all
    #               and the df_MOM should have none. Then when combined, we are fine
    #               need ot change resolve because it only deals with unique values, doesn't return full list of all records to be done
    #               OR, we need to eliminate it somewhere else
    # final_combo = pd.concat([df_MOM_with_counts, df_r], ignore_index=True)
    final_combo = pd.concat([df_MOM_with_counts, df_r], ignore_index=True)
    if False:
        print("--- final combo")
        print(final_combo.head(30))

    if False:
        n_records_87after = len(final_combo[final_combo["Burrow"] == "087"])
        print("Number of records '087' after in final COMBO: ", n_records_87after)
        records_087 = final_combo[final_combo["Burrow"] == "087"]
        print(records_087)
        print("--------")

    # 5. Sort by MOM_File then MOM_Time
    final_combo = final_combo.sort_values(
        by=["MOM_File", "MOM_Time"], ignore_index=True
    )

    # Helper: put RFID rows first (0 = has RFID, 1 = missing)
    final_combo["_rfid_rank"] = final_combo["RFID"].isna().astype(int)

    # Sort so that within each group, RFID rows are first
    final_combo = final_combo.sort_values(
        by=["Burrow", "MOM_File", "Wt", "_rfid_rank"]
    )

    # Now drop duplicates, keeping the first (which will have RFID if available)
    final_combo = final_combo.drop_duplicates(
        subset=["Burrow", "MOM_File", "Wt"],
        keep="first"
    ).reset_index(drop=True)

    # Drop helper col
    final_combo = final_combo.drop(columns="_rfid_rank")

    return final_combo

##########################
#   function: get_All_Mom_data
#       returns df with MOM data
    """
    Load all files in 'folder' that start with 'Bird_Weight_' and end with '.txt' (case-insensitive),
    combine them into a single DataFrame, remove duplicates, and return it.
    Adds a 'Burrow' column extracted from the last 3 characters of the 'File' column.
    Renames:
        'File' -> 'MOM_File'
        'Trace_Segment_Num' -> 'Segment'
    """
#######
def get_All_Mom_data(folder: str = None, one_Burr: str = None) -> pd.DataFrame:

    cols_to_import = ['File', 'Trace_Segment_Num', 'DateTime', 'Wt_Min_Slope']

    # Debug info
    # print(f"Looking in folder: {folder}")

    if folder is not None:
        folder = vTesting_Folder
    else:
        messagebox.showwarning("Find Data", "Choose folder with Mass-o-Matic data.")
        folder = filedialog.askdirectory()
        if not folder:  # User cancelled the dialog
            messagebox.showwarning("Find Data", "No Folder Selected")

    try:
        files_in_folder = os.listdir(folder)
        # print("Files in folder:")
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
                #df_temp['Burrow'] = df_temp['File'].astype(str).str[-7:-4]
                df_temp["Burrow"] = df_temp["File"].astype(str).apply(clean_burrow)
                all_dfs.append(df_temp)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not all_dfs:
        print("No matching files found.")
        return pd.DataFrame(columns=['MOM_File', 'Segment', 'DateTime', 'Wt_Min_Slope', 'Burrow'])

    # Combine and drop duplicates
    df_MOM = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    # DROP ANY ROWS WITH MISSING Wt_Min_Slope
    df_MOM["Wt_Min_Slope"] = pd.to_numeric(df_MOM["Wt_Min_Slope"], errors="coerce")
    df_MOM = df_MOM.dropna(subset=["Wt_Min_Slope"]).reset_index(drop=True)


    # Rename columns
    df_MOM.rename(columns={'File': 'MOM_File', 'Trace_Segment_Num': 'Segment'}, inplace=True)

    # Optional filter by burrow
    if one_Burr is not None:
        target = str(one_Burr).zfill(3)
        df_MOM = df_MOM[df_MOM["Burrow"] == target].reset_index(drop=True)


    return df_MOM

##########################
#   function: call_combo_RFID_MOM():
#       call from GUI to do this
#######
def do_Join_MOM_RFID(folder: str = None, one_Burr: str = None):

# global all_mom, all_rfid

    if myTesting:
        # debugging only
        folder = vTesting_Folder
    else:
        folder = None


    df_rfid = get_All_RFID_data(folder, one_Burr)
    df_WtFiles = get_All_Mom_data(folder, one_Burr)

    if df_rfid.empty or df_WtFiles.empty:
        messagebox.showwarning("Warning", "Please load both MOM and RFID data before joining.")
        return
    
    # make sure the column is actual datte/time format
    df_WtFiles['DateTime'] = pd.to_datetime(df_WtFiles['DateTime'], errors="coerce")

    # be sure we don't have any calibration data to combine
    df_WtFiles_clean = remove_spurious_pairs(df_WtFiles, "Wt_Min_Slope", low_val = 50, high_val = 75, tol  = 0.6) 

    # now link them
    df_finale= build_final_combo_MOM_RFID(df_WtFiles_clean, df_rfid)
    df_finale = df_finale.sort_values(["Burrow"], kind="mergesort").reset_index(drop=True)
    # print(df_finale.head(30))

    if True:
        # Display the joined DataFrame in t3
        join_widgetst1.delete('1.0', tk.END)  # Clear existing content
        #join_widgetst1.insert(tk.END, df_finale.to_string(index=False))  # Insert joined DataFrame

        table_str = format_df_custom(df_finale, "JOIN")
        join_widgetst1.insert(tk.END, table_str)

        # show the rfid file and the MOM files
        # rename for display because df_WtFiles_clean still uses Wt_Min_Slope
        mom_display = df_WtFiles_clean.rename(columns={"Wt_Min_Slope": "Wt"}).copy()
        mom_display["DateTime"] = pd.to_datetime(mom_display["DateTime"], errors="coerce")
        mom_display = mom_display.sort_values(["Burrow", "DateTime"])
        populate_mom_Windows(mom_display[['Burrow', 'DateTime', 'Wt']])

        rfid_display = df_rfid.copy()
        rfid_display["PIT_DateTime"] = pd.to_datetime(rfid_display["PIT_DateTime"], errors="coerce")
        rfid_display = rfid_display.sort_values(["Burrow", "PIT_DateTime"])
        populate_RFID_Windows(rfid_display[['PIT_DateTime', 'Burrow', 'Rdr', 'PIT_ID']])

        # Update label with summary counts
        try:
            df_counts = df_finale.copy()
            df_counts["Wt"] = pd.to_numeric(df_counts.get("Wt"), errors="coerce")
            total_rows = len(df_counts)
            matched_rows = df_counts[
                (df_counts["Wt"] > 0) &
                df_counts["RFID"].fillna("").astype(str).ne("")
            ]
            matched_count = len(matched_rows)
            join_widgetslabel_1.config(
                text=f"MOM Traces / RFID ({total_rows} bird weights found, {matched_count} attached to RFID)"
            )
        except Exception:
            pass


    if False:
        df_finale.to_csv("df_Final_Joined_GREAT.csv", index=False)
    else:
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
                df_finale.to_csv(output_file_path, index=False, sep="\t")
                print(f"Data saved to {output_file_path}")
            else:
                print("Save cancelled.")
        else:
            print("User chose not to save the DataFrame.")

def do_Join_One_Burrow():
    val = simpledialog.askstring("Process One Burrow", "Enter burrow number (e.g., 7 or 007):")
    if not val:
        return None # user cancelled or blank
    target = str(val).zfill(3)
    do_Join_MOM_RFID("DEBUG", target) ## change hre to DEBUG for testing folder



###########################
#   function: join_MOM_RFID2
#       Joins MOM data to nearest RFID data within a specified time window.
#       - Uses a 3-minute window by default.
#       - Tries to find RFID data shifted by -1 hour first; if no matches, tries without shift.
#       - Keeps rows in hours <07:00 or >19:00 even if no matches found. NEED PREF FOR THIS?
#       - Returns a DataFrame with columns sorted by Burrow and MOM_Time, including RFID, Rdr, and RFID_Time.
########
def join_MOM_RFID2(
    df_MOM: pd.DataFrame,
    df_RFID: pd.DataFrame,
    window: str = gTimeTolerance,
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
    hours_mask = valid_time & ((mom[mom_time_col].dt.hour < 7) | (mom[mom_time_col].dt.hour > 19)) # need a preference for this?
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
    desired_order = ["Burrow", "MOM_File", "MOM_Time", "Segmnt", "Wt", "RFID", "N", "Rdr", "RFID_Time", "RF_File"]
    out = out[[c for c in desired_order if c in out.columns]]

    # clean up data
    out["Burrow"] = out["Burrow"].astype(str).apply(clean_burrow)
    out = format_time_cols(out, date_fmt, cols=['MOM_Time', 'RFID_Time'])

    # Sort by numeric Burrow when possible, then MOM_Time
    out["Burrow_sort"] = pd.to_numeric(out["Burrow"], errors="coerce")
    out = out.sort_values(["Burrow_sort", "MOM_Time"], kind="mergesort").drop(columns=["Burrow_sort"])

    return out


def join_MOM_RFID_New(
    df_MOM: pd.DataFrame,
    df_RFID: pd.DataFrame,
    window: str = gTimeTolerance,
    mom_time_col: str = "DateTime",
    rfid_time_col: str | None = None,
) -> pd.DataFrame:
    """
    Like join_MOM_RFID2, but considers RFID shifted -1h, 0h, and +1h.
    """
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

    # Build join times (shifted -1h, zero, +1h)
    rfid["_join_time_shift"] = rfid[rfid_time_col] - pd.Timedelta(hours=1)
    rfid["_join_time_zero"]  = rfid[rfid_time_col]
    rfid["_join_time_plus"]  = rfid[rfid_time_col] + pd.Timedelta(hours=1)

    # Valid RFID per mode
    rfid_shift = rfid.dropna(subset=["_join_time_shift"])
    rfid_zero  = rfid.dropna(subset=["_join_time_zero"])
    rfid_plus  = rfid.dropna(subset=["_join_time_plus"])

    # Count matches per MOM row (try shift, then zero, then plus)
    N = np.zeros(len(mom), dtype=int)
    used_offset = np.full(len(mom), -3600, dtype=int)  # -3600 (shift) by default; 0 for zero-shift; +3600 for plus

    tol_ns = np.int64(tol.value)
    mom_times_ns = mom[mom_time_col].to_numpy(dtype="datetime64[ns]")
    mom_bkeys    = mom["_BurrowKey"].to_numpy()

    def arr_map(df: pd.DataFrame, col: str) -> dict:
        return {
            k: g[col].sort_values().to_numpy(dtype="datetime64[ns]")
            for k, g in df.groupby("_BurrowKey", sort=False)
        }

    arr_shift = arr_map(rfid_shift, "_join_time_shift")
    arr_zero  = arr_map(rfid_zero,  "_join_time_zero")
    arr_plus  = arr_map(rfid_plus,  "_join_time_plus")

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

        # Try shifted (-1h)
        arr = arr_shift.get(bk)
        c = count_in(arr, t_int) if arr is not None else 0
        if c > 0:
            N[i] = c
            continue

        # Try zero
        arr0 = arr_zero.get(bk)
        c0 = count_in(arr0, t_int) if arr0 is not None else 0
        if c0 > 0:
            N[i] = c0
            used_offset[i] = 0
            continue

        # Try plus (+1h)
        arrp = arr_plus.get(bk)
        cp = count_in(arrp, t_int) if arrp is not None else 0
        if cp > 0:
            N[i] = cp
            used_offset[i] = 3600

    mom["n_Matches"] = N

    # Keep rows in target hours (even if N==0)
    valid_time = mom[mom_time_col].notna()
    hours_mask = valid_time & ((mom[mom_time_col].dt.hour < 7) | (mom[mom_time_col].dt.hour > 19))
    reduced = mom.loc[hours_mask].copy()

    # Attach per-row offset_used (aligned by index for safety)
    reduced["_offset_used"] = pd.Series(used_offset, index=mom.index).loc[reduced.index].values

    pieces = []
    for bk, mom_g in reduced.groupby("_BurrowKey", sort=False):

        mom_shift = mom_g[mom_g["_offset_used"] == -3600].dropna(subset=[mom_time_col])
        mom_zero  = mom_g[mom_g["_offset_used"] == 0].dropna(subset=[mom_time_col])
        mom_plus  = mom_g[mom_g["_offset_used"] == 3600].dropna(subset=[mom_time_col])

        r_shift_b = rfid_shift[rfid_shift["_BurrowKey"] == bk].sort_values("_join_time_shift", kind="mergesort")
        r_zero_b  = rfid_zero [rfid_zero ["_BurrowKey"] == bk].sort_values("_join_time_zero",  kind="mergesort")
        r_plus_b  = rfid_plus [rfid_plus ["_BurrowKey"] == bk].sort_values("_join_time_plus",  kind="mergesort")

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

        if not mom_plus.empty and not r_plus_b.empty:
            m3 = pd.merge_asof(
                mom_plus.sort_values(mom_time_col, kind="mergesort"),
                r_plus_b[["_join_time_plus", "PIT_ID", "Rdr", "RF_File", "_BurrowKey"]],
                left_on=mom_time_col,
                right_on="_join_time_plus",
                by="_BurrowKey",
                direction="nearest",
                tolerance=tol,
            )
            m3["RFID_Time"] = m3["_join_time_plus"]
            merged_parts.append(m3)

        if merged_parts:
            merged_g = pd.concat(merged_parts, ignore_index=False).sort_index(kind="mergesort")
        else:
            merged_g = mom_g.copy()
            merged_g["PIT_ID"] = pd.NA
            merged_g["Rdr"] = pd.NA
            merged_g["RF_File"] = pd.NA
            merged_g["RFID_Time"] = pd.NaT

        merged_g["RFID"] = merged_g["PIT_ID"]
        pieces.append(merged_g)

    out = pd.concat(pieces, ignore_index=False).sort_index(kind="mergesort")

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

    out = out.drop(columns=["_join_time_shift", "_join_time_zero", "_join_time_plus", "PIT_ID"], errors="ignore")

    desired_order = ["Burrow", "MOM_File", "MOM_Time", "Segmnt", "Wt", "RFID", "N", "Rdr", "RFID_Time", "RF_File"]
    out = out[[c for c in desired_order if c in out.columns]]

    out["Burrow"] = out["Burrow"].astype(str).apply(clean_burrow)
    out = format_time_cols(out, date_fmt, cols=['MOM_Time', 'RFID_Time'])

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
        # print(f"Assigned variable: {name} -> {widget}")

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

# Calculate window size (80% of screen width) - changed to 0.5 for testing on big screen - detect screen size?
window_width = int(screen_width * 0.5)
window_height = int(screen_height * 0.5)

root.geometry(f"{window_width}x{window_height}")

##########################
# Create a parent container for RFID + MOM buttons side-by-side
##########################
if Show_Buttons:
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
if Show_Buttons:
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
join_widgetsframe = join_widgets["join_widgetsframe"]
join_widgetst1 = join_widgets["join_widgetst1"]  # explicit for linters    

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
if do_print:
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

# Process menu mirrors the Join RFID+MOM Data button
process_menu = tk.Menu(menubar, tearoff=False)
menubar.add_cascade(label="Process", menu=process_menu)
# process_menu.add_command(label="Join BSM/RFID", command=do_Join_MOM_RFID)

process_menu.add_command(label="Join GPS/RFID (Debug)", command=lambda: do_Join_MOM_RFID("DEBUG"))
process_menu.add_command(label="Process One Burrow",command=do_Join_One_Burrow)


##########################
# Start the app going wiht the mainloop
#######
root.mainloop()
