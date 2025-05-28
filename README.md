# XRD Refinement Plot Application

This project is a Dash application designed to plot X-ray diffraction (XRD) data exported from **TOPAS Academic**. It provides a user-friendly interface for uploading TOPAS `.txt` and HKL files and visualizing the results in a graph.

**Deployed via Heroku.**  
Access the app at: _[Heroku app URL here]_

## Project Structure

```
dash-xrd-app
├── app.py               # Main Dash application file
├── demo_folder          # Example TOPAS Academic files for demo/testing
│   ├── GdRu0.25Ge2-hkl
│   ├── GdRu0.25Ge2.txt
│   └── Gd3Ru4Ge13-hkl
├── requirements.txt     # List of dependencies
└── README.md            # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```
   cd dash-xrd-app
   ```
3. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

## Running the Application Locally

To run the Dash application locally, execute the following command in your terminal:
```
python app.py
```
This will start the server, and you can access the application in your web browser at `http://127.0.0.1:8050`.

## Input File Formats

**All files must be exported from TOPAS Academic.**

### HKL File Format

Each HKL file should be a whitespace-delimited text file, for example:
```
   9.845592    0.000000
  13.941008  188.767276
  17.095435   51.212299
  19.764818    0.000000
  22.125555    0.000000
  24.268012    0.000009
  28.093879   93.494947
  29.836426   22.919037
  29.836426   22.919037
  31.491083  101.702958
  33.071147   46.821843
```

### `.txt` Data File Format

The `.txt` file should be a CSV exported from TOPAS Academic, with a header and columns for x, y, Ycalc, and Diff. Example:
```
'y-axis saved as y
'x,216-1hr_Gd-Ru-Ge.xy,Ycalc,Diff
9.99786,1000,1015.32001,-15.3200087
10.00816,912,1014.31659,-102.31659
10.01847,905,1013.31351,-108.313507
10.02878,935,1012.31174,-77.3117363
10.03909,871,1011.31128,-140.311278
...
```

## Demo Data

A `demo_folder` is included with example files:
- `GdRu0.25Ge2-hkl`
- `GdRu0.25Ge2.txt`
- `Gd3Ru4Ge13-hkl`

You can use these files to test the application.

## Plot Title

The plot title is automatically generated based on the uploaded `.txt` file name (for example, uploading `GdRu0.25Ge2.txt` will result in the title:  
**GdRu0.25Ge2 refinement plot**)
Numbers and dots in the title are automatically subscripted.

## Usage

1. Upload your TOPAS `.txt` data file and one or more HKL files using the file upload component in the application.
2. The application will process the files and display the plotted data in the graph.
3. You can interact with the graph and explore the data visually.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.