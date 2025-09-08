# Instructions To Run:

## Pre-requisites:

- Python must be installed on the system, version 3.12 is ideal to guarantee the code runs

## 1. Clone/(download and unzip) the project

Ensure it is beside the folder containing the dataset and that the dataset folder is named 'dataset'. The videos may be in nested subfolders.
```
    dataset/
    project/
```

## 2. Do the following in the command line:

### a. `cd` into the project folder

### b. Create a virtual environment

Windows: `python -m venv .venv`

Mac/Linux: `python3 -m venv .venv`

- Note: for this step, if multiple versions of Python are installed on the system, it may be necessary to replace `python` with the appropriate command to invoke the correct version of Python

### c. Activate the virtual environment

Windows: `.venv\Scripts\activate.bat`

Mac/Linux: `source env/bin/activate`

### d. Install the required packages inside the environment

Windows/Mac/Linux: `pip install -r requirements.txt`

### e. Run the main.py file with Python

Windows: `python main.py`

Mac/Linux: `python3 main.py`

The output is stored in a folder called 'output', beside 'src' and 'dataset', in the following format:

```
    output/
        changepoints.csv
        1_frames/
            video1/
                0.png
            video2/
                0.png
            ...
        2_frames/
            video1/
                0.png
                1.png
            video2_0/
                0.png
                1.png
            ...
        3_frames/
            video1/
                0.png
                1.png
                2.png
            video2_0/
                0.png
                1.png
                2.png
            ...
```

### f. Deactivate the virtual environment

Windows/Mac/Linux: `deactivate`

---

> Note: When rerunning the program, the virtual environment need not be recreated (skip step 2.b and 2.d). Rerunning the program will overwrite the existing contents of the output folder.
