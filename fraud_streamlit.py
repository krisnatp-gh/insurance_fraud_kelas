import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .fraud-detected {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .no-fraud {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained XGBoost model pipeline"""
    try:
        with open('insurance_fraud_model.sav', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'insurance_fraud_model.sav' not found. Please ensure the file is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_input_features():
    """Create input widgets for all features"""
    
    st.markdown('<h1 class="main-header">🔍 Insurance Fraud Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Temporal Information")
        month = st.selectbox("Month", 
                            ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug', 'Jul', 'May', 'Sep'])
        
        day_of_week = st.selectbox("Day of Week", 
                                  ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'])
        
        week_of_month = st.slider("Week of Month", 1, 5, 3)
        week_of_month_claimed = st.slider("Week of Month (Claimed)", 0, 5, 3)
        
        month_claimed = st.selectbox("Month Claimed", 
                                    ['Jan', 'Nov', 'Jul', 'Feb', 'Mar', 'Dec', 'Apr', 'Aug', 'May', 'Jun', 'Sep', 'Oct', '0'])
        
        day_of_week_claimed = st.selectbox("Day of Week (Claimed)", 
                                          ['Tuesday', 'Monday', 'Thursday', 'Friday', 'Wednesday', 'Saturday', 'Sunday', '0'])
        
    with col2:
        st.subheader("🚗 Vehicle & Policy Information")
        make = st.selectbox("Vehicle Make", 
                           ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 
                            'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 
                            'Mercedes', 'Ferrari', 'Lexus'])
        
        vehicle_category = st.selectbox("Vehicle Category", ['Sport', 'Utility', 'Sedan'])
        
        vehicle_price = st.selectbox("Vehicle Price Range",
                                    ['more than 69000', '20000 to 29000', '30000 to 39000', 'less than 20000',
                                     '40000 to 59000', '60000 to 69000'])
        
        age_of_vehicle = st.selectbox("Age of Vehicle",
                                     ['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new', '4 years', '2 years'])
        
        policy_type = st.selectbox("Policy Type",
                                  ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils',
                                   'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability',
                                   'Sport - All Perils'])
        
        base_policy = st.selectbox("Base Policy", ['Liability', 'Collision', 'All Perils'])
        
        deductible = st.slider("Deductible", 300, 700, 400, step=100)
        
    with col3:
        st.subheader("👤 Personal Information")
        sex = st.selectbox("Sex", ['Female', 'Male'])
        
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Widow', 'Divorced'])
        
        age = st.slider("Age", 16, 80, 40)
        
        age_of_policy_holder = st.selectbox("Age Group of Policy Holder",
                                           ['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25',
                                            '36 to 40', '16 to 17', 'over 65', '18 to 20'])
        
        fault = st.selectbox("At Fault", ['Policy Holder', 'Third Party'])
        
        accident_area = st.selectbox("Accident Area", ['Urban', 'Rural'])
        
        driver_rating = st.slider("Driver Rating", 1, 4, 2)
        
        # Additional features in a separate section
        st.subheader("📊 Additional Details")
        
        policy_number = st.number_input("Policy Number", min_value=1, max_value=15420, value=7710)
        rep_number = st.slider("Rep Number", 1, 16, 8)
        
        year = st.slider("Year", 1994, 1996, 1995)
        
        days_policy_accident = st.selectbox("Days Policy to Accident",
                                           ['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'])
        
        days_policy_claim = st.selectbox("Days Policy to Claim",
                                        ['more than 30', '15 to 30', '8 to 15', 'none'])
        
        past_number_of_claims = st.selectbox("Past Number of Claims",
                                            ['none', '1', '2 to 4', 'more than 4'])
        
        police_report_filed = st.selectbox("Police Report Filed", ['No', 'Yes'])
        witness_present = st.selectbox("Witness Present", ['No', 'Yes'])
        agent_type = st.selectbox("Agent Type", ['External', 'Internal'])
        
        number_of_suppliments = st.selectbox("Number of Suppliments",
                                            ['none', 'more than 5', '3 to 5', '1 to 2'])
        
        address_change_claim = st.selectbox("Address Change Claim",
                                           ['1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months'])
        
        number_of_cars = st.selectbox("Number of Cars",
                                     ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'])
        
    return {
        'Month': month,
        'DayOfWeek': day_of_week,
        'Make': make,
        'AccidentArea': accident_area,
        'DayOfWeekClaimed': day_of_week_claimed,
        'MonthClaimed': month_claimed,
        'WeekOfMonth': week_of_month,
        'WeekOfMonthClaimed': week_of_month_claimed,
        'Sex': sex,
        'MaritalStatus': marital_status,
        'Age': age,
        'Fault': fault,
        'PolicyType': policy_type,
        'VehicleCategory': vehicle_category,
        'VehiclePrice': vehicle_price,
        'Days_Policy_Accident': days_policy_accident,
        'Days_Policy_Claim': days_policy_claim,
        'PastNumberOfClaims': past_number_of_claims,
        'AgeOfVehicle': age_of_vehicle,
        'AgeOfPolicyHolder': age_of_policy_holder,
        'PoliceReportFiled': police_report_filed,
        'WitnessPresent': witness_present,
        'AgentType': agent_type,
        'NumberOfSuppliments': number_of_suppliments,
        'AddressChange_Claim': address_change_claim,
        'NumberOfCars': number_of_cars,
        'BasePolicy': base_policy,
        'PolicyNumber': policy_number,
        'RepNumber': rep_number,
        'Deductible': deductible,
        'DriverRating': driver_rating,
        'Year': year
    }

def make_prediction(model, features):
    """Make fraud prediction using the trained model"""
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Create sidebar with model info
    with st.sidebar:
        st.markdown("### 📋 Model Information")
        st.markdown("""
        <div class="info-box">
        <strong>Model:</strong> XGBoost Classifier<br>
        <strong>Purpose:</strong> Insurance Fraud Detection<br>
        <strong>Output:</strong> Binary Classification (0: No Fraud, 1: Fraud)<br>
        <strong>Features:</strong> 29 input variables
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Instructions")
        st.markdown("""
        1. Fill in all the required information
        2. Click 'Predict Fraud Risk' to get results
        3. Review the prediction and confidence score
        4. Use results to guide investigation priorities
        """)
    
    # Get input features
    features = create_input_features()
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing claim data..."):
                prediction, prediction_proba = make_prediction(model, features)
                
                if prediction is not None:
                    # Display prediction
                    if prediction == 1:
                        st.markdown("""
                        <div class="prediction-box fraud-detected">
                            ⚠️ FRAUD DETECTED ⚠️<br>
                            This claim shows high risk indicators
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-box no-fraud">
                            ✅ LOW FRAUD RISK<br>
                            This claim appears legitimate
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display confidence scores
                    st.markdown("### 📊 Confidence Scores")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("No Fraud Probability", f"{prediction_proba[0]:.3f}", 
                                 help="Probability that this claim is legitimate")
                    
                    with col2:
                        st.metric("Fraud Probability", f"{prediction_proba[1]:.3f}",
                                 help="Probability that this claim is fraudulent")
                    
                    # Progress bars for visual representation
                    st.markdown("### 📈 Risk Assessment")
                    st.progress(float(prediction_proba[1]), text=f"Fraud Risk: {prediction_proba[1]:.1%}")
                    
                    # Risk interpretation
                    fraud_prob = prediction_proba[1]
                    if fraud_prob >= 0.7:
                        risk_level = "🔴 HIGH RISK"
                        recommendation = "Immediate investigation recommended"
                    elif fraud_prob >= 0.4:
                        risk_level = "🟡 MEDIUM RISK"
                        recommendation = "Additional review suggested"
                    else:
                        risk_level = "🟢 LOW RISK"
                        recommendation = "Standard processing appropriate"
                    
                    st.markdown(f"""
                    **Risk Level:** {risk_level}  
                    **Recommendation:** {recommendation}
                    """)
    
    # Add footer with educational information
    st.markdown("---")
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Model Details
        This XGBoost model was trained to detect potentially fraudulent insurance claims based on various factors including:
        
        - **Temporal patterns**: When accidents and claims occurred
        - **Vehicle information**: Make, model, age, and value
        - **Personal details**: Demographics and policy holder information  
        - **Claim characteristics**: Policy details and claim circumstances
        - **Historical data**: Past claims and policy changes
        
        ### Important Notes
        - This is a predictive model and should be used as a tool to assist human judgment
        - High-risk predictions warrant further investigation, not automatic rejection
        - Model predictions should be combined with domain expertise and additional evidence
        - Regular model retraining is recommended as fraud patterns evolve
        
        ### For Educational Use
        This application demonstrates practical ML model deployment using Streamlit,
        showcasing real-world application of classification algorithms in insurance.
        """)

if __name__ == "__main__":
    main()