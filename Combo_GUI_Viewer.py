#######################################
#######################################
#    MOM_GUI.py
#       GUI elements and execution 
#       R.A.M and L.U.T.
#       2024-08-27 cleanup of RAM_v10
#       CHANGES: 9/1/2024
#           Add button for batch auto processing
#           Add function def: setup_gui() to properlyl handle GUI
#       CHANGES: 12/6/2024
#           Change Auto button to perform Duration calculations - change in list of button names and call to method
#######################################
#######################################

VERSION = "v12(2024-12-07)"

#######################################
#######################################
# Imports and libraries
#######################################
#######################################

import tkinter as tk
import MOM_Processing
import MOM_Calculations

#######################################
#######################################
# GUI Parameters
#######################################
#######################################

# Program screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Three subpanels: Inputs, Buttons, Outputs

# Input frame
INPUT_PAD_X = 30
INPUT_PAD_Y = 20

# Button frames
BUTTON_PAD_X = 5
BUTTON_PAD_Y = 5
BUTTON_WIDTH = 20

# Which interface to use
use_Duration = False

if use_Duration:
    BUTTON_LABELS = ["View", "Manual", "Duration", "Auto Batch"]
else:
    BUTTON_LABELS = ["View", "Manual", "Automatic", "Auto Batch"]

# Output frames
OUTPUT_FRAME_WIDTH = 400
OUTPUT_FRAME_HEIGHT = 300
OUTPUT_TEXT_PAD_X = 50
OUTPUT_TEXT_PAD_Y = 25

def setup_gui():
    # Initialize GUI
    # Root window
    root = tk.Tk()
    root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    root.title("Mass-O-Matic Analyzer " + VERSION)

    # Initialize calibrations
    calibration = MOM_Calculations.Calibration()

    # Input frame for user-defined calibration values 
    input_frame = tk.Frame(root, width=SCREEN_WIDTH, bd=1, padx=INPUT_PAD_X, pady=INPUT_PAD_Y, relief=tk.FLAT)
    input_frame.pack()
    initial_calibration_labels = ["Calib. mass light:", "Calib. mass med:", "Calib. mass heavy:"]
    initial_calibration_values = calibration.get_true()

    # Initialize the input frames, filled by default with the default-initialized true calibration values
    # NOTE these will change automatic if you change the default values in the Calibration constructor
    calibration_user_entered_values = []
    for i, label in enumerate(initial_calibration_labels):
        tk.Label(input_frame, text=initial_calibration_labels[i]).grid(row=0, column=i, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
        entry = tk.Entry(input_frame)
        entry.grid(row=1, column=i, padx=5, pady=5)
        entry.insert(0, initial_calibration_values[i])
        calibration_user_entered_values.append(entry)

    # Buttons
    button_frame = tk.Frame(root, width=BUTTON_WIDTH-10, bd=0, relief=tk.SOLID)
    button_frame.pack(pady=5)

    # Place buttons in frames
    # Each button is associated with a core function in MOM_Processing
    buttons = []

    # View button calls MOM_Processing.view()
    button_view = tk.Button(button_frame, text=BUTTON_LABELS[0], command=lambda: MOM_Processing.view(output_frame_text))
    button_view.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_view)

    # Manual button calls MOM_Processing.process_manual()
    button_manual = tk.Button(button_frame, text=BUTTON_LABELS[1], command=lambda: MOM_Processing.process_manual(calibration, calibration_user_entered_values, output_frame_text))
    button_manual.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_manual)

    if not use_Duration:
        # Auto button calls MOM_Processing.process_auto()
        button_auto = tk.Button(button_frame, text=BUTTON_LABELS[2], command=lambda: MOM_Processing.process_auto(calibration, calibration_user_entered_values, output_frame_text))
    else:
        # Auto button now calls MOM_Processing.view("duration")
        button_auto = tk.Button(button_frame, text=BUTTON_LABELS[2], command=lambda: MOM_Processing.view(output_frame_text, "duration"))

    button_auto.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_auto)

    # Auto Batch button calls MOM_Processing.process_auto_batch_2() 
    button_auto_batch = tk.Button(button_frame, text=BUTTON_LABELS[3], command=lambda: MOM_Processing.process_auto_start(calibration, calibration_user_entered_values, output_frame_text, show_graph=True))
    button_auto_batch.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_auto_batch)

    # Add a Trim button - calls MOM_Processing.trim()- NEW FEATURE
    button_trim = tk.Button(button_frame, text="Trim", command=lambda: MOM_Processing.trim())
    button_trim.pack(side=tk.LEFT, padx=BUTTON_PAD_X, pady=BUTTON_PAD_Y)
    buttons.append(button_trim)


    # Output frame with text widget
    output_frame = tk.Frame(root, width=OUTPUT_FRAME_WIDTH, height=OUTPUT_FRAME_HEIGHT, bd=0, relief=tk.SOLID)
    output_frame.pack(padx=OUTPUT_TEXT_PAD_X, pady=OUTPUT_TEXT_PAD_Y)

    # Vertical scrollbar for text widget
    output_scrollbar = tk.Scrollbar(output_frame, orient="vertical")
    output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Place text widgets in frame
    output_frame_text = tk.Text(output_frame, width=OUTPUT_FRAME_WIDTH, height=OUTPUT_FRAME_HEIGHT, yscrollcommand=output_scrollbar.set)
    output_frame_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    output_scrollbar.config(command=output_frame_text.yview)

    # Output text widget cannot be configured by user
    # NOTE configure back to "normal" state before writing output from program
    #      this is usually performed within output_() functions
    output_frame_text.configure(state="disabled")

    return root


#############
# Added to let GUI present information correctly
#####
if __name__ == "__main__":
    root = setup_gui()
    root.mainloop()
