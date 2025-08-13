################
#
#   DAD_Viewer_v2.0 by RAMauck, June 28, 2025
#       built from v11_GPS_accel frorm June 2025 with help from ChatGPT
#
#       New
#           implement new meta data export
#           implement new details for orgainzing data
#       NOTE: need to check VeDBA calcs agaoinst: MOESM1_ESM_R_code_Behav.pdf
#
###########

##########################
#   Import libraries needed
#       tkinter for interface
#       pandas for database management
#       matplotlib for the graphing
#       numpy for the stats, etc.       
########
import tkinter as tk
from tkinter import messagebox, filedialog, Menu, simpledialog, Scrollbar
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates  # Make sure to import this

##########################
#   Include other local files needed for the app to work and globals they need
########
# from GPS_Tools_v03 import handle_import_and_join_gps
from GPS_Tools_v03 import load_and_display_gps_coords
from GPS_Tools_v03 import join_GPS_Accel
from GPS_Tools_v03 import show_gps_map
from GPS_Tools_v03 import summarize_trips  # import the data function

from BS_Utility_FNs import extract_tag_from_filename

# from Accel_Tools_v01 to handle all the accelerometer data
from Accel_Tools_v01 import load_accelerometer_file
from Accel_Tools_v01 import make_hourly_vedba_summary   # This function generates the hourly VeDBA summary to show the user after loading

import DAD_Globals

global_accel_filename = ""
DAD_Globals.curr_tag_global = ""
global mean_vedba

# Define globally available variables
dataframe = pd.DataFrame() 
date_fmt = '%Y %m %d %H %M %S.%f'  # Format string for datetime conversion
day_only_fmt = '%m-%d-%Y'

global vVersString
global vAppName
vVersString = " (v_2.4)"  # implements WBF calculations, corrects VeDBA calcs, and adds GPS data handling
vAppName = "DAD Viewer" + vVersString


##########################
#   load_file
#       Current wrapper file used to load all teh Accel ldata from PathTrak
#       Need to change name to 'load_file_accel'
#       loads accelerometry data, parses the data, summarizes them for the output
#       assumes Pathtrak file format 
#       NOTE: PathTrak calcs a column they call "VeDBA" which is only the sum of the 3 axes squared  
#           We calculate a our own VeDBA column which is more complicated - from ChatGPT
#           Need to check literature for proper method currently being used      
########
def load_file():
    global dataframe, burst_summary_df_global, global_accel_filename

    f_types = [('CSV files', "*.csv"), ('TXT', "*.txt"), ('CSV files', "*.CSV"), ('TXT', "*.TXT")]
    file_path = filedialog.askopenfilename(title="Choose Accelerometer File", filetypes=f_types)
    if not file_path:
        return

    try:
        print("in DAD Viewer - load the accel file")
        # Load data, burst summary, and tag name
        dataframe, burst_summary_df, tag, prefix = load_accelerometer_file(file_path)

        print("in DAD Viewer - show burst_summary_df")
        print(burst_summary_df.head(3))

        # Store global metadata
        global_accel_filename = os.path.basename(file_path)
        DAD_Globals.curr_tag_global = tag
        DAD_Globals.curr_prefix_global = prefix
        burst_summary_df_global = burst_summary_df
        DAD_Globals.mean_vedba_df_global = dataframe

        # Generate hourly VeDBA summary for plotting
        mean_vedba_df = make_hourly_vedba_summary(burst_summary_df)

        # Display burst summary in text widget
        display_df = burst_summary_df[['Center_Time', 'Mean_VeDBA']]
        t1.delete('1.0', tk.END)
        t1.insert('end', display_df.to_string(index=False))

        # Plot summary and update UI
        process_and_plot(mean_vedba_df)

        record_count = len(burst_summary_df_global)
        label_all_records.config(text=f"{DAD_Globals.curr_tag_global}: VeDBA Averaged Bursts ({record_count})")

        if DAD_Globals.gps_df_global is not None:
            record_count2 = len(DAD_Globals.gps_df_global)
            label_gps_coords.config(text=f"{DAD_Globals.curr_tag_global}: GPS Coordinates ({record_count2})")
        else:
            label_gps_coords.config(text=f"{DAD_Globals.curr_tag_global}: GPS Coordinates")

        update_button_states()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")



##########################
#   export_hourly_summary
#       uses already loaded accel data to create a simple text file for export
#       now called by a button       
########
def export_hourly_summary():
    from tkinter import filedialog

    if 'mean_vedba_df_global' not in globals() or dataframe is None:
        messagebox.showerror("Export Error", "No summary available. Please load a file first.")
        return

    mean_vedba_df = mean_vedba_df_global

    # Get the first timestamp in each (Date, Hour) group
    burst_starts = (
        dataframe
        .reset_index()
        .groupby([dataframe.index.date, dataframe.index.hour])
        .agg(Start_Time=('DateTime', 'min'))
        .reset_index(drop=True)
    )

    # Center timestamp = start + 60 seconds
    burst_starts['Center_Time'] = burst_starts['Start_Time'] + pd.Timedelta(seconds=60)

    # Combine with VeDBA summary
    export_df = mean_vedba_df.copy()
    export_df['Center_Time'] = burst_starts['Center_Time'].values

        # Use tag in default filename
    filename_default = f"{DAD_Globals.curr_tag_global}_Hourly_Summary.csv"

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        initialfile=filename_default,
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
        title="Save Hourly VeDBA Summary"
    )

    if file_path:
        export_df.to_csv(file_path, index=False)
        messagebox.showinfo("Export Complete", f"Hourly summary exported to:\n{file_path}")

##########################
#   export_burst_summary
#       similar to export_hourly_summary
#       but exports a summary of each burst, not summarized over the hour
#       could combine these two methods and pass a paraemter for which to use      
########
def export_burst_summary(export_csv=True):
    from tkinter import filedialog

    if 'dataframe' not in globals() or dataframe is None or 'BurstID' not in dataframe.columns:
        messagebox.showerror("Export Error", "No burst data available. Please load a file first.")
        return None, None

    # Reattach index if needed
    df = dataframe.reset_index()

    # Get one summary row per burst
    burst_summary = df.groupby('BurstID').agg(
        Center_Time=('DateTime', lambda x: x.min() + pd.Timedelta(seconds=60)),
        VeDBA=('VeDBA', 'mean'),
        N=('VeDBA', 'count')
        # ODBA=('ODBA', 'mean'),
        # N=('ODBA', 'count')
    ).reset_index(drop=True)

    file_path = None
    if export_csv:
        # Use tag in default filename
        filename_default = f"{DAD_Globals.curr_tag_global}_BurstSummary.csv"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=filename_default,
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
            title="Save Burst-Level VeDBA Summary"
        )

        if file_path:
            burst_summary.to_csv(file_path, index=False)
            messagebox.showinfo("Export Complete", f"Burst summary exported to:\n{file_path}")

    return burst_summary, file_path

##########################
#   process_and_plot
#       basic tool to put different aspects of accel data on screen      
########
def process_and_plot(df):
    print("DataFrame columns:", df.columns)
    
    if 'VeDBA' not in df.columns:
        raise ValueError("DataFrame must contain 'Mean_VeDBA' column")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(df.index, df['VeDBA'], marker='o', linestyle='-', color='b')
    ax.set_title('Hourly Mean VeDBA Over Time')
    ax.set_xlabel('Index')
    ax.set_ylabel('Mean VeDBA')

    # Clear previous plots in the top frame
    for widget in frame_graph_top.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_graph_top)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

##########################
#   plot_Z
#       plot the Z data from the XYZ accel points
#       right now it uses a global for the accel dataframe
#       future: pass the df to make it more flexible?      
########
def plot_Z():
    global dataframe
    
    if 'Z' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'Z' column")
    
    if dataframe['Z'].empty:
        raise ValueError("The 'Z' column is empty")

    fig, ax2 = plt.subplots(figsize=(6, 4), dpi=100)
    
    ax2.plot(dataframe.index, dataframe['Z'], marker='o', linestyle='-', color='r')
    ax2.set_title('Z Values Over Time')
    ax2.set_xlabel('DateTime')
    ax2.set_ylabel('Z Value')

     # Format x-axis to show only Month-Day
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))


    # Clear previous plots in the bottom frame
    for widget in frame_graph_bottom.winfo_children():
        widget.destroy()

    canvas2 = FigureCanvasTkAgg(fig, master=frame_graph_bottom)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)


##########################
#   plot_Axis
#       Wrapper to plot any accel axis from opened Pathtrak file
#       called by buttons for plotting       
########
def plot_Axis(axis):
    if axis not in ['X', 'Y', 'Z', 'VeDBA']:
        raise ValueError("Invalid axis. Please choose 'X', 'Y', 'Z', or 'VeDBA'.")
    
    plt.close('all')  # Close all previous figures so that we don't get confused

    #setup and format the graph widget
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(dataframe.index, dataframe[axis], marker='o', linestyle='-', color='g')
    ax.set_title(f'{axis} Values Over Time')
    ax.set_xlabel('DateTime')
    ax.set_ylabel(f'{axis} Value')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)

    # Enable interactive mode and put it on screen
    plt.ion()  
    plt.show()

#####################################
#   Below here is all setting up the UI
#   Should be method to call, or it's own file, like the MOM deal
#   In fact, should have a GUI.py file that can be called included in any project, then customized
###################

#################
#   update_button_states
#       UI feature to activate buttons only when appropriate
#####
def update_button_states():
    if DAD_Globals.mean_vedba_df_global is not None:
        button_plot_Z.config(state=tk.NORMAL)
        button_plot_X.config(state=tk.NORMAL)
        button_plot_Y.config(state=tk.NORMAL)
        button_plot_VeDBA.config(state=tk.NORMAL)
        button_export_summary.config(state=tk.NORMAL)
        button_export_bursts.config(state=tk.NORMAL)
    else:
        button_plot_Z.config(state=tk.DISABLED)
        button_plot_X.config(state=tk.DISABLED)
        button_plot_Y.config(state=tk.DISABLED)
        button_plot_VeDBA.config(state=tk.DISABLED)
        button_export_summary.config(state=tk.DISABLED)
        button_export_bursts.config(state=tk.DISABLED)

    if DAD_Globals.gps_df_global is not None:
        print("gps_df loaded")
        #btn_show_map.config(state=tk.NORMAL)
    else:
        print("gps_df NOT loaded")
        #btn_show_map.config(state=tk.DISABLED)
    
    if DAD_Globals.mean_vedba_df_global is not None and DAD_Globals.gps_df_global is not None:
        print("both df are full")
        button_join.config(state=tk.NORMAL)
    else:
        print("both df are NOT full")
        button_join.config(state=tk.DISABLED)


##########################
#   display_df_to_frame
#       UI for sending text to thelower frame- i.e.k results of trip summary       
########
def display_df_to_frame(df, text_widget):
    text_widget.delete('1.0', tk.END)
    text_widget.insert(tk.END, df.to_string(index=False))


root = tk.Tk()
root.title(vAppName)

# Set window size to 90% width and full height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width * 0.9)
window_height = screen_height
root.geometry(f"{window_width}x{window_height}")

# ---------- TOP: Button Row ----------
frame_buttons_date = tk.Frame(root)
frame_buttons_date.pack(side=tk.TOP, fill=tk.X, padx=20, pady=20)

# Buttons for data loading and plotting
button_load_file = tk.Button(frame_buttons_date, text="Load Accel", command=load_file)
button_load_file.pack(side=tk.LEFT)

button_plot_Z = tk.Button(frame_buttons_date, text="Plot Z", command=lambda: plot_Axis('Z'), state=tk.DISABLED)
button_plot_Z.pack(side=tk.LEFT, padx=10)

button_plot_X = tk.Button(frame_buttons_date, text="Plot X", command=lambda: plot_Axis('X'), state=tk.DISABLED)
button_plot_X.pack(side=tk.LEFT, padx=10)

button_plot_Y = tk.Button(frame_buttons_date, text="Plot Y", command=lambda: plot_Axis('Y'), state=tk.DISABLED)
button_plot_Y.pack(side=tk.LEFT, padx=10)

button_plot_VeDBA = tk.Button(frame_buttons_date, text="Plot VeDBA", command=lambda: plot_Axis('VeDBA'), state=tk.DISABLED)
button_plot_VeDBA.pack(side=tk.LEFT, padx=10)

button_export_summary = tk.Button(frame_buttons_date, text="Expt hourly VeDBA", command=export_hourly_summary, state=tk.DISABLED)
button_export_summary.pack(side=tk.LEFT, padx=10)

button_export_bursts = tk.Button(frame_buttons_date, text="Expt Burst VeDBA", command=export_burst_summary, state=tk.DISABLED)
button_export_bursts.pack(side=tk.LEFT, padx=10)

button_join = tk.Button(frame_buttons_date, text="Join Accel/GPS",
    command=lambda: join_GPS_Accel(DAD_Globals.mean_vedba_df_global, burst_summary_df_global,
                                    do_export=True, show_summary_fn=display_df_to_frame,
                                    summary_widget=text_trip_summary),
    state=tk.DISABLED)
button_join.pack(side=tk.LEFT, padx=10)

button_load_gps_coords = tk.Button(frame_buttons_date, text="Load GPS",
    command=lambda: load_and_display_gps_coords(gps_output_text, label_gps_coords, update_button_states))
button_load_gps_coords.pack(side=tk.LEFT, padx=10)

# ---------- CENTER: Main Content Layout ----------
mainContentFrame = tk.Frame(root)
mainContentFrame.pack(fill=tk.BOTH, expand=True)

# -------- LEFT: Output Frame (Bursts) --------
outputFrame = tk.Frame(mainContentFrame, width=300, bd=1, relief=tk.SOLID)
outputFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

label_all_records = tk.Label(outputFrame, text="VeDBA Averaged Bursts", font=("Arial", 12))
label_all_records.pack(pady=(10, 5))

frame_t1 = tk.Frame(outputFrame, bd=1, relief=tk.SOLID)
frame_t1.pack(pady=5, fill=tk.BOTH, expand=True)

scrollbar1 = Scrollbar(frame_t1, orient="vertical")
scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)

myOutputWidth = 50
myOutputHeight = 20

t1 = tk.Text(frame_t1, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar1.set)
t1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar1.config(command=t1.yview)

# -------- CENTER: Graph Frame --------
graphFrame = tk.Frame(mainContentFrame, width=600, bd=1, relief=tk.SOLID)
graphFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_graph_top = tk.Frame(graphFrame, width=600, height=250, bd=1, relief=tk.SOLID, bg='lightgrey')
frame_graph_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_graph_bottom = tk.Frame(graphFrame, width=600, height=250, bd=1, relief=tk.SOLID, bg='lightgrey')
frame_graph_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

scrollbar_summary = Scrollbar(frame_graph_bottom)
scrollbar_summary.pack(side=tk.RIGHT, fill=tk.Y)

text_trip_summary = tk.Text(frame_graph_bottom, wrap=tk.NONE, yscrollcommand=scrollbar_summary.set, bg='white')
text_trip_summary.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_summary.config(command=text_trip_summary.yview)

# -------- RIGHT: GPS Frame --------
gpsFrame = tk.Frame(mainContentFrame, width=300, bd=1, relief=tk.SOLID)
gpsFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

label_gps_coords = tk.Label(gpsFrame, text="GPS Coordinates", font=("Arial", 12))
label_gps_coords.pack(pady=(10, 5))

frame_gps_text = tk.Frame(gpsFrame, bd=1, relief=tk.SOLID)
frame_gps_text.pack(pady=5, fill=tk.BOTH, expand=True)

scrollbar_gps = Scrollbar(frame_gps_text, orient="vertical")
scrollbar_gps.pack(side=tk.RIGHT, fill=tk.Y)

gps_output_text = tk.Text(frame_gps_text, width=myOutputWidth, height=myOutputHeight, yscrollcommand=scrollbar_gps.set)
gps_output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_gps.config(command=gps_output_text.yview)


def quit_app():
    root.quit()

#################
# don't want a menu for now
#####
if(False):
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    file_menu = tk.Menu(menubar, tearoff=False)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Quit", command=quit_app)

##########################
#   Let's get started       
########
root.mainloop()
