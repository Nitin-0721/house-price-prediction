import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")
st.markdown("### Predict the price of a house based on its features!")
st.markdown("---")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('../models/best_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    # Load ORIGINAL unscaled data for correct medians
    df = pd.read_csv('../data/train.csv')
    df = df.drop(df[(df['GrLivArea'] > 4000) & 
                    (df['SalePrice'] < 200000)].index)
    return model, scaler, df

model, scaler, df_original = load_model()

# Input form
st.sidebar.header("🏡 Enter House Features")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Living Area (sqft)", 500, 6000, 1500)
garage_cars = st.sidebar.slider("Garage Cars Capacity", 0, 4, 2)
garage_area = st.sidebar.number_input("Garage Area (sqft)", 0, 1500, 400)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sqft)", 0, 3000, 800)
first_flr_sf = st.sidebar.number_input("1st Floor Area (sqft)", 500, 4000, 1000)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
year_built = st.sidebar.number_input("Year Built", 1900, 2024, 2000)
yr_sold = st.sidebar.number_input("Year Sold", 2006, 2024, 2010)
tot_rms_abv_grd = st.sidebar.slider("Total Rooms", 2, 14, 6)

if st.sidebar.button("🔮 Predict Price"):
    # Load X_train column names
    feature_cols = pd.read_csv('../data/X_train.csv').columns.tolist()
    
    # Get median from ORIGINAL unscaled data
    df_num = df_original.select_dtypes(include=np.number)
    
    # Build input using original medians
    input_dict = {}
    for col in feature_cols:
        if col in df_num.columns:
            input_dict[col] = df_num[col].median()
        else:
            input_dict[col] = 0
    
    # Override with user values
    input_dict['OverallQual'] = overall_qual
    input_dict['GrLivArea'] = gr_liv_area
    input_dict['GarageCars'] = garage_cars
    input_dict['GarageArea'] = garage_area
    input_dict['TotalBsmtSF'] = total_bsmt_sf
    input_dict['1stFlrSF'] = first_flr_sf
    input_dict['FullBath'] = full_bath
    input_dict['YearBuilt'] = year_built
    input_dict['YrSold'] = yr_sold
    input_dict['TotRmsAbvGrd'] = tot_rms_abv_grd
    input_dict['TotalSF'] = total_bsmt_sf + first_flr_sf
    input_dict['HouseAge'] = yr_sold - year_built
    input_dict['RemodAge'] = yr_sold - year_built
    input_dict['TotalBath'] = full_bath
    input_dict['HasGarage'] = 1 if garage_area > 0 else 0
    input_dict['HasBasement'] = 1 if total_bsmt_sf > 0 else 0
    input_dict['HasPool'] = 0

    # Create dataframe
    input_df = pd.DataFrame([input_dict])

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    log_price = model.predict(input_scaled)[0]
    predicted_price = np.expm1(log_price)

    # Show results
    st.markdown("## 🎯 Prediction Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price", f"${predicted_price:,.0f}")
    with col2:
        st.metric("Overall Quality", f"{overall_qual}/10")
    with col3:
        st.metric("Living Area", f"{gr_liv_area} sqft")

    st.success(f"✅ Estimated House Price: **${predicted_price:,.0f}**")

    lower = predicted_price * 0.90
    upper = predicted_price * 1.10
    st.info(f"📊 Price Range: **${lower:,.0f}** to **${upper:,.0f}** (±10%)")

    st.markdown("### 📊 House Features Summary")
    summary = {
        'Feature': ['Overall Quality', 'Living Area', 'Garage Cars',
                   'Basement Area', 'Full Bathrooms', 'Year Built'],
        'Value': [f"{overall_qual}/10", f"{gr_liv_area} sqft",
                 garage_cars, f"{total_bsmt_sf} sqft",
                 full_bath, year_built]
    }
    st.table(pd.DataFrame(summary))

else:
    st.markdown("## 👈 Enter house features in the sidebar and click Predict!")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🤖 **Model:** Ridge Regression")
        st.info("📈 **Accuracy:** 90% (R² = 0.9006)")
    with col2:
        st.info("📊 **Training Data:** 1166 houses")
        st.info("🏠 **Dataset:** Ames Housing")