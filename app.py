#import necessary libraries
import pickle
import pandas as pd
import numpy as np
import streamlit as st 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image
import shap
from random import randint 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
    
    

# Loading models
model = pickle.load(open("./Pickle Saved Data/ML_Model1.pkl", 'rb'))
model2 = pickle.load(open("./Pickle Saved Data/random_forest_model.pkl", 'rb'))


# Initialize session state for storing risk level
if 'risk_level' not in st.session_state:
    st.session_state['risk_level'] = None

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Banking Prediction System', ['Financial Risk Prediction', 'Loan Prediction', 'Dashboard', 'About'], icons=['person', 'activity', '','building'], default_index=0)

# Financial Risk Prediction Page
if(selected == 'Financial Risk Prediction'):
    st.title('Financial Risk Prediction')
    st.markdown('Predict the financial risk of a person based on their financial statements.')

    # Uploading the CSV file
    st.header('Upload Data')
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    # Data and Model Parameters
    st.header('Set Parameters for  Model Configuration')
    split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    model_params = {
        'n_estimators': st.slider('Number of estimators', 10, 1000, 100),
        'random_state': st.slider('Random state', 0, 1000, 42),
    }
    st.header('Risk calculation')
    # Function to build and evaluate the model
    def build_model(df, split_size, model_params):
        X = df.iloc[:,:-1]  # Features
        Y = df.iloc[:,-1]   # Target
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100, random_state=model_params['random_state'])

        rf = RandomForestRegressor(**model_params)
        rf.fit(X_train, Y_train)

        # Displaying model performance and details
        st.subheader('Model Performance')
        for set_name, X_set, Y_set in [("Training", X_train, Y_train), ("Testing", X_test, Y_test)]:
            Y_pred = rf.predict(X_set)
            st.markdown(f'**{set_name} set R^2:**')
            st.info(r2_score(Y_set, Y_pred))
            st.markdown(f'**{set_name} set Error (MSE):**')
            st.info(mean_squared_error(Y_set, Y_pred))
            st.subheader('Model Parameters')
            st.write(rf.get_params())

####
    def predict_risk():
   
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.number_input('City Area Code from [0-44]', min_value=0, max_value=44, value=1)
            location_score = st.number_input('Location Score from [0-100]', min_value=0.00, max_value=100.00, value=10.00)
        with col2:
            internal_audit_score = st.number_input('Internal Audit Score from [0-15]', min_value=0, max_value=15, value=1)
            external_audit_score = st.number_input('External Audit Score from [0-15]', min_value=0, max_value=15, value=1)
        with col3:
            fin_score = st.number_input('Financial Score from [0-15]', min_value=0, max_value=15, value=1)
            loss_score = st.number_input('Loss Score from [0-13]', min_value=0, max_value=13, value=1)
            past_results = st.slider('Past Results', 0, 10, 1)

        if st.button('Risk Prediction'):
            features = [[city, location_score, internal_audit_score, external_audit_score, fin_score, loss_score, past_results]]
            prediction = model2.predict(features)
            st.session_state['risk_level'] = int(prediction[0])  # Store the risk prediction
            if prediction == 0:
                st.markdown('The financial risk of this person is **Low**.')
            else:
                st.markdown('The financial risk of this person is **High**.')
                # Main panel display
            st.subheader('Dataset Overview')
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(df.head())
                build_model(df, split_size, model_params)
            else:
                st.info('Awaiting for CSV file to be uploaded.')

    predict_risk()

# Loan Prediction Page
if(selected == 'Loan Prediction'):
    st.title(" Loan Prediction")


    def run():
        img1 = Image.open('Bank.png')
        img1 = img1.resize((450,245))
        st.image(img1, use_column_width=False)

        # Check if risk level has been predicted and display it
        if st.session_state['risk_level'] is None:
            st.warning("Please perform the Financial Risk Prediction first.")
            st.stop()
        else:
            # Display the current risk level
            risk_level = "High" if st.session_state['risk_level'] else "Low"
            st.write(f"Based on your Financial Risk Prediction, your risk level is: **{risk_level}**.")

        # Account No
        account_no = st.text_input('Account number')

        # Full Name
        fn = st.text_input('Full Name')

        # Gender
        gen_display = ['Female', 'Male']
        gen_options = list(range(len(gen_display)))
        gen = st.selectbox("Gender", gen_options, format_func=lambda x: gen_display[x])

        # Marital Status
        mar_display = ['No', 'Yes']
        mar_options = list(range(len(mar_display)))
        mar = st.selectbox("Marital Status", mar_options, format_func=lambda x: mar_display[x])

        # Dependents
        dep_display = ['No', 'One', 'Two', 'More than Two']
        dep_options = list(range(len(dep_display)))
        dep = st.selectbox("Dependents", dep_options, format_func=lambda x: dep_display[x])

        # Education
        edu_display = ['Not Graduate', 'Graduate']
        edu_options = list(range(len(edu_display)))
        edu = st.selectbox("Education", edu_options, format_func=lambda x: edu_display[x])

        # Employment Status
        emp_display = ['Job', 'Business']
        emp_options = list(range(len(emp_display)))
        emp = st.selectbox("Employment Status", emp_options, format_func=lambda x: emp_display[x])

        # Property Area
        prop_display = ['Rural', 'Semi-Urban', 'Urban']
        prop_options = list(range(len(prop_display)))
        prop = st.selectbox("Property Area", prop_options, format_func=lambda x: prop_display[x])

        # Credit Score
        cred_display = ['Between 300 to 500', 'Above 500']
        cred_options = list(range(len(cred_display)))
        cred = st.selectbox("Credit Score", cred_options, format_func=lambda x: cred_display[x])

        # Monthly Income
        mon_income = st.number_input("Applicant's Monthly Income($)", value=0)

        # Co-Applicant Monthly Income
        co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", value=0)

        # Loan Amount
        loan_amt = st.number_input("Loan Amount", value=0)

        # Loan Duration
        dur_display = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
        dur_options = range(len(dur_display))
        dur = st.selectbox("Loan Duration", dur_options, format_func=lambda x: dur_display[x])

        if st.button("Submit"):
            duration = [60, 180, 240, 360, 480][dur]
            features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
            prediction = model.predict(features)
            ans = int("".join(str(i) for i in prediction))
            if ans == 0:
                st.error(f"Hello: {fn} || Account number: {account_no} || According to our calculations, you will not get the loan from the bank.")
            else:
                st.success(f"Hello: {fn} || Account number: {account_no} || Congratulations!! You will get the loan from the bank.")

            if risk_level == "High":
                st.warning("Note: Your high risk level may affect the loan approval process.")
            # Continue with prediction and display results
    run()

#calling from previous Part1_Prediction(RiskPrediction) for checking distribution
train_data = pd.read_csv("Risk_Dataset/Train.csv")
X1= train_data.drop(['IsUnderRisk'], axis=1)
y1= train_data['IsUnderRisk']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X1,y1 , test_size=0.2 ,random_state= 42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
mod1 = lr.fit(X_train,y_train)

###calling from previous Part1_Prediction(LoanPrediction) for checking distribution
train1=pd.read_csv('./Loan_Dataset/train.csv')
label_encoder = LabelEncoder()
for column in train1.columns:
    if train1[column].dtype == 'object':
        train1[column] = label_encoder.fit_transform(train1[column])
Loan_status=train1.Loan_Status
train1.drop("Loan_Status", axis=1, inplace=True)
train1.Credit_History.fillna(np.random.randint(0,2),inplace=True)
train1.Married.fillna(np.random.randint(0,2),inplace=True)
train1.LoanAmount.fillna(train1.LoanAmount.median(),inplace=True)
train1.Loan_Amount_Term.fillna(train1.Loan_Amount_Term.mean(),inplace=True)
train1.Gender.fillna(np.random.randint(0,2),inplace=True)
train1.Dependents.fillna(train1.Dependents.median(),inplace=True)
train1.Self_Employed.fillna(np.random.randint(0,2),inplace=True)
X=train1.iloc[:614,] ## all the data in X (Train set)
y=Loan_status  ## Loan status will be our Y
from sklearn.ensemble import ExtraTreesClassifier
mod2 = ExtraTreesClassifier()
mod2.fit(X,y)

#Dashboard Option
if(selected == 'Dashboard'):
    st.title('Dashboard')
    st.markdown('Statistical Analysis of financial statement Data')
    def dash():

            # Assuming train_data is your DataFrame
            fig = px.histogram(train_data)

            # Update layout to adjust figure size
            fig.update_layout(width=800, height=600)  # Adjust width and height according to your preference

            # Display the plot in Streamlit
            st.plotly_chart(fig)
            
            # Create a SHAP explainer object using the trained model
            explainer = shap.Explainer(mod1, X_train)  # Using X_train for creating the explainer

            # Compute SHAP values for the test data
            shap_values = explainer(X_test)

            # Plot SHAP summary plot within a new matplotlib figure context
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)  # Ensure show=False to capture plot without displaying it immediately
            st.pyplot(plt.gcf())  # Get the current figure and pass it to Streamlit
             # Create a Series of feature importances
            feat_importances = pd.Series(mod2.feature_importances_, index=X.columns)

            st.markdown('**Feature Analysis of Loan Data**')

            # Plot graph of feature importance for better visualization
            plt.figure(figsize=(10, 8))  # You can adjust the size as needed
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')

            # Display the plot in Streamlit
            st.pyplot(plt.gcf()) 

           
    dash()

#About Option
if(selected == 'About'):
            st.write("**Welcome to the Loan and Risk Prediction App!**")
            st.write("**Developer: Sathwik Reddy Chelemela**")
            st.write("License: MIT License")
            st.code("""
            Copyright (c) 2024 Sathwik Reddy Chelemela
            Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
            The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
            """)
            st.title('About Our Banking Prediction System')
            st.header('Empowering Financial Decisions with Predictive Analytics')
            st.write('''
            In the intricate world of financial services, assessing risk and determining 
            loan eligibility are not just transactions—they are critical decisions that shape 
            the futures of individuals and the stability of financial institutions. Our Banking 
            Prediction System is engineered to bridge the gap between financial analysis and 
            predictive foresight, ensuring that every lending decision is backed by data-driven 
            confidence.
            ''')

            st.subheader('Financial Risk Prediction: The First Step')
            st.write('''
            Before delving into the realm of loan prediction, our platform prioritizes understanding 
            the financial risk associated with a potential borrower. Risk prediction is the cornerstone 
            of sound financial decision-making. By analyzing a spectrum of financial statements and 
            scores, we ascertain the risk profile of a person or entity. This evaluation is not just 
            a number; it's a comprehensive assessment that encapsulates the likelihood of default, 
            predicting financial health with precision that traditional methods cannot match.
            ''')

            st.subheader('The Connection Between Risk and Loan Eligibility')
            st.write('''
            The correlation between financial risk and loan eligibility is pivotal. A lower risk 
            score indicates a higher probability of timely repayment, translating into more favorable 
            loan terms. Conversely, a higher risk profile might necessitate a more cautious lending 
            approach. Our system ensures that risk assessment is not an isolated metric but a dynamic 
            part of the lending lifecycle that influences every subsequent decision.
            ''')

            st.subheader('Loan Prediction: Beyond Eligibility')
            st.write('''
            Once the financial risk is assessed, the focus shifts to loan prediction. This stage is 
            about determining not if, but how a loan should be structured. Loan prediction is an art 
            form that balances financial risk, customer needs, and long-term profitability. It's about 
            crafting solutions that are sustainable for both lender and borrower, paving the way for 
            financial success and growth.
            ''')

            st.subheader("From an Employer's Perspective")
            st.write('''
            For employers, our system offers an unprecedented level of insight. It allows for an evaluation 
            of financial proposals with a clear understanding of potential risks and returns. Employers can 
            use our platform to make informed decisions that are aligned with their financial strategies and 
            risk appetite, ensuring that every investment in human capital is a step toward organizational 
            resilience.
            ''')

            st.subheader('Complexity Simplified')
            st.write('''
            The complexity of our work lies in transforming a maze of data into clear predictive insights. We 
            harness the power of advanced analytics, machine learning algorithms, and user-friendly interfaces 
            to provide a service that is as complex in its calculations as it is simple in its presentation. 
            Our commitment is to uncomplicate the complicated, making sophisticated financial predictions 
            accessible to all.
            ''')

            st.subheader('The Use Case: A Spectrum of Scenarios')
            st.write('''
            Our Banking Prediction System is versatile, catering to a wide range of use cases from individual 
            credit assessments to large-scale corporate financial analyses. Whether it's a small personal loan 
            or a significant commercial investment, our system provides the predictive clarity needed to move 
            forward with confidence.
            ''')

            st.write('Join us as we redefine financial prediction, merging analytics with acumen to empower every financial decision.')

            st.write("This web application is designed to assist with financial decision-making by predicting loan approvals and assessing financial risk.")
            st.write("For any inquiries or feedback, please contact Sathwik Reddy Chelemela at chelemela.s@northeastern.edu")
            st.write("Connect with me on : [LinkedIn](https://www.linkedin.com/in/sathwikreddychelemela/)")
            st.write("Check out my profile on : [Github ](https://github.com/SathwikReddyChelemela?tab=repositories)")
            st.write("Check out Video Explanation : [Youtube](https://youtu.be/pGj6FCgKczc)")
