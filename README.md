# AA894-Group-4-Capstone
Repository for PSU AI 894 Capstone Project

Team:
- Raphael San Andres
- Johnny Zielinski
- Vincenzo Marquez

Abstract: This repository contains the source code, visualizations, dataset, and other important files regarding the building and creation of an AI model that can predict an NFL formation based on the positions of the players and their relative x and y coordinates.


# Installation and Execution

- Ensure you update `requirements.txt` when adding new dependencies.
- Use a virtual environment to avoid conflicts with global Python packages.

Running the *local_venv_setup.sh* script will automatically setup your virtual environment which includes all necessary packages to replicate this project. The command below is an example command for Windows to run the script:

```
# Assuming you are in the root directory of the project
./local_venv_setup.sh
```

# Individual Models
## AutoGluon
To run AutoGluon use the notebooks found under **src/models/autogluon**. This includes all necessary source code for AutoGluon specific testing, results, and visualizations.
## H2O.ai
To run H2O.ai use the notebooks found under src/models/h2oAI. This includes all necessary source code for H2O specific testing, results, and visualizations.
## XGBoost
To run XGBoost use the notebooks found under src/models/XGBoost. This includes all necessary source code for H2O specific testing, results, and visualizations.

# Adhoc Scripts
Several adhoc scripts for task like data consolidation, expoloration, and visualization can be found under **src/notebooks** these can be run individually for the separate tasks listed in their titles.