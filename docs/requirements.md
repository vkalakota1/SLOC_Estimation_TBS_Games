# Project Requirements and Predictors

**Objective:**  
Develop a robust SLOC estimation tool for open-source turn-based strategy games.

**Predictors:**  
- **Traditional Predictors:**  
  - **NRUL:** Number of gameplay rules.
  - **MGOP:** Miscellaneous game options (menu items, buttons).
  - **NPLY:** Number of players.
  - **ANIM:** Animation complexity.
- **Domain-Specific Predictors:**  
  - **MAP:** Number of maps (levels, environments).
  - **UNU:** Number of unique unit types.
  - **NP:** Number of phases per turn (for layered decision-making).

**System Architecture Overview:**  
The system is divided into modules for data collection, preprocessing, feature extraction, model building (using forward stepwise regression), validation (using K-fold cross-validation), and deployment with interactive what-if analysis.
