
# README for Financial Risk and Loan Prediction Project


Video Explanation : https://youtu.be/pGj6FCgKczc

## Overview
This project comprises two main parts: predicting financial risk and determining loan eligibility. It uses machine learning models to analyze various factors and predict outcomes, which are then integrated into a web application to provide real-time predictions.

## Contents
1. **Part1_Prediction.ipynb**: Jupyter notebook detailing the approach to calculate financial risk based on features such as city, location score, audit scores, and more. The notebook uses a dataset from Kaggle for this purpose.
2. **Part2_Prediction.ipynb**: Jupyter notebook focused on predicting whether customers will receive a loan from a bank. This prediction considers factors like gender, education, marital status, loan amount, and credit history.
3. **app.py**: Python script for a Streamlit web application that interfaces with the models developed in the Jupyter notebooks to provide a user-friendly platform for real-time predictions.
4. **requirements.txt**: Lists all the necessary Python libraries needed to ensure compatibility and functionality across the tools used in this project.

## Usage
### Part1: Financial Risk Prediction
- **Dataset**: Financial risk prediction data from Kaggle.
- **Features**: Includes city, location score, internal and external audit scores, financial score, loss score, past results, and risk indicator.

### Part2: Loan Eligibility Prediction
- **Dataset**: Loan eligibility data from Kaggle, provided by Dream Housing Finance.
- **Features**: Gender, marital status, education, dependents, income, loan amount, credit history, account balance, property area, etc.

### Application
- **Functionality**: The application allows users to input their data and receive predictions on financial risk and loan eligibility.
- **Libraries Used**: Streamlit for the web app, Pandas and NumPy for data handling, Matplotlib for visualization, and Pickle for loading models.
## Advanced Data Analysis Techniques

### Part1_Prediction.ipynb: Financial Risk Prediction
- **VIF (Variance Inflation Factor)**:  to detect multicollinearity among features such as location score and audit scores.
- **OLS (Ordinary Least Squares)**:  used for regression analysis to understand relationships between scores if applicable.
- **Feature Importance**:  to Analyze which features most strongly predict financial risk, refining features for better model performance.

### Part2_Prediction.ipynb: Loan Eligibility Prediction
- **Tree Node Interpretability**:  tree-based models is used to help understand decision-making at each node.
- **SHAP (SHapley Additive exPlanations) Analysis**: To show the impact of each feature on the prediction of loan eligibility, enhancing model transparency.
- **Feature Importance**: Determining which features are most predictive of loan eligibility, informing feature prioritization.


## Installation and Setup
- Install required Python libraries using pip. The exact versions and libraries are specified in the `requirements.txt`:
  pip install -r requirements.txt
- Run the notebooks to train the models or review the data analysis.
- Launch the Streamlit application: streamlit run app.py
  

## Future Enhancements
- Improve the predictive models with more comprehensive data.
- Enhance the user interface of the web application for better user experience.
- Implement additional features based on user feedback.

For more information on each part or technical details, refer to the comments and documentation within the notebooks and Python script.
