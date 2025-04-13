# SLOC Estimation for Turn-Based Strategy Games

**Project Overview:**  
This project develops a domain-specific tool to estimate the Source Lines of Code (SLOC) in open-source turn-based strategy games. It integrates both traditional predictors (e.g., game rules, players, animations) and domain-specific predictors (e.g., number of maps, unique unit types, phases per turn) through a robust regression model.

**Key Features:**  
- Data collection and preprocessing from open-source game repositories.
- Feature extraction and integrated predictor analysis.
- Model building using forward stepwise multiple linear regression.
- Validation using K-fold cross-validation.
- A web-based interface with interactive what-if analysis.

**Installation and Setup:**  
See [requirements.md](docs/requirements.md) for detailed setup instructions.

**License:**  
This project is released under the MIT License.
