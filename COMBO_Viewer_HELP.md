# COMBO Viewer Help

Version: v_02.4b

## What COMBO Viewer does

COMBO Viewer links Mass-o-Matic (MOM) trace data with RFID detections by time and burrow.
It supports:

- Manual loading and manual line-by-line joins.
- Automatic processing for all burrows or one burrow.
- Saving combined/manual output.

## Input files

### RFID files

- File name pattern: starts with `RF` and ends with `.txt`.
- Expected columns (no header row): `PIT_ID`, `Rdr`, `PIT_DateTime`.

### MOM files

- File name pattern: starts with `Bird_Weight_` and ends with `.txt`.
- Expected columns (header row): `File`, `Trace_Segment_Num`, `DateTime`, `Wt_Min_Slope`.

## Menus

### File

- `Quit`: exits COMBO Viewer.

### Process Manual

- `Load RFID Files: All`: choose folder and load all RFID files.
- `Load RFID Files: One Burrow`: choose folder and load only one burrow.
- `Load MOM Files: All`: choose folder and load all MOM files.
- `Load MOM Files: One Burrow`: choose folder and load one burrow.
- `Join Manually`: enables manual action buttons between Traces and combined output.

### Process Automatic

- `Join GPS/RFID`: run automatic MOM + RFID matching.
- `Process One Burrow`: run automatic processing for one burrow.

### Help

- `View Help`: opens this help file in a scrollable window.

## Manual join workflow

1. Load RFID and MOM data from `Process Manual`.
2. Choose `Process Manual -> Join Manually` to activate buttons.
3. Select one line in `RFIDs`.
4. Select one line in `Traces`.
5. Use buttons:
   - `Join`: append combined row to bottom of `MOM Traces / RFID`.
   - `Insert`: insert combined row above selected line in output (or append if none selected).
   - `Remove`: remove selected output line.
   - `Clear All`: clear output area.
   - `Save`: save output (including header) to a text file.

## Selection behavior

- In all three output panes, selection is limited to one line at a time.
- Header lines stay fixed; only body rows scroll.

## Troubleshooting

- If no data appears after loading:
  - confirm folder contains matching file names.
  - confirm file columns are in expected format.
- If manual join warns about missing selection:
  - select one line in `RFIDs` and one line in `Traces`.
- If Save warns there is no output:
  - create at least one manual joined line first.

## Notes

- Manual output save is currently fixed-width text format.
- CSV/TAB/fixed output preferences can be added later.
