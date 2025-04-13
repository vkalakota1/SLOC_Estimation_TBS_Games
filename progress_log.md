# Progress Log: Sprint 1 (Requirements, Planning, and Environment Setup)

**Tasks Completed:**
- Created GitHub repository "SLOC_Estimation_TBS_Games" with initial README, .gitignore, and LICENSE.
- Organized project directory structure: backend, frontend, data, docs, appendix.
- Documented project requirements and finalized predictors in docs/requirements.md.
- Set up the Python virtual environment and installed Flask, numpy, pandas, scikit-learn, and graphviz.
- Created preliminary system architecture diagram (docs/system_architecture_diagram.png).
- Committed and pushed the initial setup to GitHub.

# Progress Log - Sprint 2: Data Collection, Preprocessing, and Feature Extraction

**Completed Tasks:**
- Created `game_data.csv` in the /data folder with realistic sample data.
- Developed and executed `preprocess_data.py` to normalize the data and remove outliers. The cleaned data is saved as `game_data_processed.csv`.
- Developed and executed `extract_features.py` to extract and display predictor values from the processed dataset.

**Next Steps:**
- Explore automating the feature extraction directly from game source code.
- Expand the dataset by gathering additional data from more open-source turn-based strategy games.
- Proceed to Sprint 3: Model Building and Validation.
