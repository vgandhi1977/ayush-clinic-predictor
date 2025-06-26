import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
import unicodedata
import json

# Utility to normalize strings for safe comparison
def normalize(s):
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode().lower().strip()

# Load GeoJSON
with open("ndap_data/india_states.geojson", "r") as f:
    geojson_data = json.load(f)

st.header("🗺️ AYUSH Clinics Across Indian States")

# Load cleaned AYUSH dataset
map_df = pd.read_csv("ndap_data/AYUSH_Merged_Cleaned.csv")

# Normalize & map state names
map_df["StateName"] = map_df["StateName"].str.lower().str.strip()

# Map to match GeoJSON
state_name_mapping = {
    "andaman and nicobar islands": "Andaman and Nicobar",
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jammu and kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Orissa",
    "puducherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttaranchal",
    "west bengal": "West Bengal",
    "the dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu"
}

map_df["StateName"] = map_df["StateName"].replace(state_name_mapping)
map_df["StateName_clean"] = map_df["StateName"].apply(normalize)

# Normalize GeoJSON state names
geojson_states = {normalize(f["properties"]["name"]) for f in geojson_data["features"]}

# Show mismatches
mismatched = set(map_df["StateName_clean"]) - geojson_states
if mismatched:
    st.warning(f"🚫 Still mismatched: {sorted(mismatched)}")
else:
    st.success("✅ All states match GeoJSON successfully!")

# Plot choropleth
#map_df["StateName"] = map_df["StateName"].str.title()  # Match display format
fig = px.choropleth(
    map_df,
    geojson=geojson_data,
    featureidkey="properties.name",
    locations="StateName",
    color="Total_AYUSH",
    color_continuous_scale="YlGnBu",
    title="State-wise AYUSH Clinic Count"
)
fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, use_container_width=True)

# ------------------- MODEL PREDICTION SECTION -------------------

# Load trained model
model = joblib.load("ayush_model.pkl")

st.title("🧪 AYUSH Clinic Predictor")
uploaded_file = st.file_uploader("📁 Upload your input CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Preview of Uploaded Data")
        st.write(df.head())

        features = ['TotalPopulation', 'TotalPopulationUrban', 'TotalPopulationRural',
                    'LandArea', 'LandAreaUrban', 'LandAreaRural', 'NumberOfHouseholds']

        if all(f in df.columns for f in features):
            X = df[features]
            predictions = model.predict(X)
            df['Predicted_AYUSH_Clinics'] = predictions

            st.subheader("📊 Predicted AYUSH Clinics")
            st.write(df[['StateName', 'Predicted_AYUSH_Clinics']])
        else:
            st.error("🚫 Missing required columns in uploaded CSV.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Please upload a CSV to begin.")

