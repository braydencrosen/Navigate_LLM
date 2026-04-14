# Navigate_LLM

Navigate_LLM is a research tool created for **Project Navigate** at the *University of Florida's Public Utility Research Center*, designed to analyze semantic change in a large language model's output. There are 2 files designed to be run by the user: `main.py` is used to run simulation events, `computation.py` is used to run computations on the previously generated output data.

The program works by first selecting a random article, which is sent to 4 configured personas of an LLM. The LLM is asked to respond, and the output is saved and embedded.

Once the program has selected 4 random articles, (one from each specified topic), the process will run again, and will select 4 different random articles, prefaced by an intervention article.

This process is named 1 event. 

After at least 1 event has been completed, computations can be run on that event. The program will compute numerous cosine similarities and euclidean distances within the output vectors for that event. Similarities can be saved or displayed as requested.

The project is configured to accept 2 `.csv` files as article databases. To function properly, each article must have a unique identifier (even across databases).

Prompting the LLM is done entirely by the program, however, prompt variables must first be set in `config.py`. There are 4 variables that must be set (not including API key or base URL) to prompt as intended:

- PREFACE - This is the text that will start the call, and can be used to describe the project.

- PERSONAS - This is where personas can be defined

- INTERVENTION_PREFACE - This is the preface that will precede the intervention article in the AFTER phase

- INTERVENTION - This is the RAW intervention article text, that will precede the standard article in the AFTER phase

Fake articles have been added to the repository, which can clarify formatting needs for real data. Please note that these article databases are **not** real and have been AI-generated.

## Disclaimer

This repository is the public version of this project, which does not include the real article databases, configuration, or research data that was used to compute data for **Project Navigate** at the *University of Florida*. All real articles and configuration variables have been reset or removed.

# Setup for MacOS

To ensure setup accuracy, verify that the project folder name is exactly `Navigate_LLM`

### Install Python

If you have already installed Python 3.11 or greater, you can skip this step. Otherwise, you can install it [here](https://www.python.org/downloads/). Once Python has been installed, open terminal and enter the following command

```bash
python3 --version
```

### Change to the project directory

Open terminal and change to the Navigate_LLM directory

```bash
cd ~/.../Navigate_LLM
```

For example, if the Navigate_LLM folder is on your desktop:

```bash
cd ~/Desktop/Navigate_LLM
```

If this command works, you may continue. If not:

- Enter `cd` and drag the project folder into the terminal. The path should appear.
- Press `enter`. 
- Remember this path for future reference, as you will need it to run the project.

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

### Install Dependencies

Before running this command, ensure you are working inside of the project directory, and that your virtual environment is activated. You should see something like 

```bash
(.venv) yourusername@Your-Device-Name Navigate_LLM %
```
If you see this, you are ready to continue. Otherwise, restart the process.

The following command may take a few moments to complete, as your computer will begin installing the required dependencies into the virtual environment.

```bash
python3 -m pip install -r requirements.txt
```

### Set your API key

Before running the code, you must configure `API_BASE_URL` and `MODEL_NAME`. The model name should be input as your exact model as a string, and the base URL can be found in your API dashboard.

The model you choose must be compatible with Open AI's Python module.

Once you have obtained your API key, copy it to your clipboard and paste it in this command between quotation marks. **Keep this key secure and do not share it with anyone.** 

This command will store the key permanently for the project.

```bash
echo 'export API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

If you only want to save your key for **this session**, enter this command instead

```bash
export API_KEY="your_api_key_here"
```

### Run the program

Confirm you are working inside of the **Navigate_LLM** directory with your virtual environment activated and your API key set. If these steps have been completed, you can now run the project entering the following command

```bash
python3 main.py
```

## MacOS Run Instructions

Assuming all setup instructions have been followed and you are just starting from a *new session*, you can enter this command inside of Terminal to prepare to run the project. Enter the project folder's path in place of `...`

```bash
cd ~/.../Navigate_LLM
source .venv/bin/activate
```

Once the above steps have been followed, you should see something like

```bash
(.venv) yourusername@Your-Device-Name Navigate_LLM %
```

Once you confirm your terminal shows this, you can proceed with running the program.

# Setup For Windows

To ensure setup accuracy, verify that the project folder name is exactly `Navigate_LLM`

### Install Python

If you have already installed Python 3.11 or greater, you can skip this step. Otherwise, you can install it [here](https://www.python.org/downloads/). Once Python has been installed, open terminal and enter the following command

### Change to the project directory

Open command prompt and change to the Navigate_LLM directory

```bash
cd C:\...\Navigate_LLM
```

For example, if the Navigate_LLM folder is on the desktop:

```bash
cd C:\Users\YourUsername\Desktop\Navigate_LLM
```

### Create and activate a virtual environment

```bash
py -3 -m venv .venv
.venv\Scripts\activate
```

After running these commands, you should see something like

```bash
(.venv) C:\Users\YourUsername\...\Navigate_LLM>
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Set API key
To persist your key permanently for future sessions, enter

```bash
setx API_KEY "your_api_key_here"
```

To set this for your *current session only*, enter

```bash
set API_KEY=your_api_key_here
```

### Run the program

Confirm you are working inside of the project's directory with your virtual environment activated and your API key set. If these steps have been completed, you can now run the project entering the following command

```bash
python3 main.py
```

## Windows run instructions

Assuming all setup instructions have been completed and you are simply running the program in a new session, open a command prompt window and enter the following commands

```bash
cd C:\...\Navigate_LLM
.venv\Scripts\activate
```

Once this has been completed, you should see something like

```bash
(.venv) C:\Users\YourUsername\...\Navigate_LLM
```

Once this is confirmed in your command prompt window, you may proceed with running the program.

# Running Navigate_LLM

Do not proceed to this step until the OS-specific setup and run instructions have been followed.

### Configuration

This project currently allows the user to use models compatible with the OpenAI client in Python.

#### Configuring your model
1. Obtain your API key and set it as an environment variable using the setup instructions above
2. Open **Config.py** and
3. Set `MODEL_NAME` to your exact model name as a string
4. Set `API_BASE_URL` to your exact base url as a string

### Simulation

Running a simulation will allow you to select a database source and number of events. Once these variables are set, your computer will begin by 
* Prompting the model
* Saving output
* Vectorizing output

You can run simulations by entering 

```bash
python3 main.py
```

### Computation

Once you have sucessfully completed at least one simulation event, you are ready to run a computation. This program will allow you to select
* Single event computation
* Cross-event computation
* Custom vector computation
* Batch computation (on all data in active VECTORS file)

Each of which will calculate the cosine similarities and euclidean distances of a specific combination of vectors.

- The batch computation will also include a summary file which includes averages on all data computed.

Once a computation is complete, you will have the option to display it, entering `n` for the *display prompt* will **not** erase the computation data. 

Once you have selected a display option, you will then be prompted to save the file. Files can be saved as either .txt or .pdf, and will contain the exact output text that displays if you choose to display the output.

*If you choose **not** to save the data, you will be asked to confirm your choice. If you again choose not to save the computation data, it will be **permanetly erased**. However, you can obtain this same data by running the same computation again, given that the vectors file has not been tampered with*

You can run computations by entering

```bash
python3 computation.py
```

By default, saved computation data will be saved in a folder named "Computation_Data" within the project's directory, though this folder name can be reconfigured in *config.py*

# Troubleshooting

```bash
no output
```
If the project has been running for more than 2 minutes, and you don't see any output to the terminal, it is likely that the vectorization model is failing to connect. 

- Check your network connection
- Check you are working in the (venv)
- Check you are in the correct directory
- Terminate the process by entering `CTRL+C` and try running again

<br>

```
An error occured, check log.jsonl for more information. Terminating session.
```
The project had a runtime error and the exception was logged. You can open log.jsonl and view the error from the project folder using any text application. **Visual Studio Code** is recommended, though **Text Edit** or **Notepad** will work as well. You will see the error thrown in the log.

<br>

```bash
Model terms of service violated, Output evt___-__ will not be vectorized.If this issue persists, consider changing prompt.
```

The language model's terms of service have been violated, and it responded with an automated fault message. This error will not affect surrounding data. To prevent computation inaccuracies, a vector will not be created for this output. When running a computation for this event, you will see that the computation associated with this output number will display `NO COMPARISON GENERATED`, which is due to a missing vector.

If this issue persists across multiple events, it is possible that a certain article or persona could be an issue. If so, consider removing or editing the cause.

<br>

```
_______ exception logged: disregard evt___ output __ vector, do not use it for any calculations.
```

An exception was thrown for this specific prompt try, if it continues to occur, it is likely that the model is misconfigured and is not responding.

If this error happens once, and is not reoccuring, you can disregard it.

<br>

```bash
One or more computations could not be completed due to missing vector data
```

The computation process was completed, but the program is indicating that some computations were not able to be completed due to missing data. This is not a fatal error, and the computations in the generated data can be trusted, it is likely that vectors were not generated due to a Terms of Service violation or a fatal error during simulation.

# Removing Data

In the event you may want to remove all event data:

- Run the main program
- When prompted to select a database, enter `clr`
- Enter `y`

Please note, this change is **permanent** and cannot be undone. Removing data by this process will delete all generated event output and all event vectors by removing the following files:

- event_vectors.jsonl
- events.csv

This will **not** remove computation data, however, it is highly recommended that you export the computation folder to avoid event mismatch, which could lead to data misinterpretation. You can simply do this by dragging the folder to another location outside of the Navigate_LLM directory. After doing this, saving a computation will automatically create a new folder.

It is recommended to use this process to **remove** saved data if necessary, as it prevents any critical files from accidentally being deleted.

This feature exists for **test** purposes, as it allows the user to quickly remove generated data.

If you want to reset data, and also keep the data that has been generated, simply move them to a folder and name it as needed. The program will not append files unless they have the exact name as specified in the `config.py` file, so renaming the files works as well.

---
### Contact

For any setup, licensing, or troubleshooting questions / concerns, please contact me at [braydencrosen@outlook.com](mailto:braydencrosen@outlook.com).