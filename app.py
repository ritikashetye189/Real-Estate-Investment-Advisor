
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    classification_model = joblib.load(
        BASE_DIR / "models" / "classification_model.pkl"
    )
    regression_model = joblib.load(
        BASE_DIR / "models" / "regression_model.pkl"
    )
    return classification_model, regression_model

st.title("🏡 Real Estate Investment Advisor")
st.write("Enter property details to explore the model's predictions.")

try:
    classification_model, regression_model = load_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

def choose(label, options):
    return st.selectbox(label, options)

with st.form("property_form"):
    st.subheader("Property Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        state = choose("State", [
            "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
            "Uttar Pradesh", "Telangana", "Kerala", "Other"
        ])
        city = st.text_input("City", "Mumbai")
        locality = st.text_input("Locality", "Locality_1")
        property_type = choose("Property Type", [
            "Apartment", "Independent House", "Villa"
        ])
        bhk = st.number_input("BHK", 1, 10, 2)
        size = st.number_input("Size (sq. ft.)", 100, 20000, 1000, step=50)
        price = st.number_input(
            "Current Price (₹ lakhs)", 1.0, 10000.0, 100.0, step=5.0
        )
        year_built = st.number_input("Year Built", 1950, 2026, 2020)

    with c2:
        furnished = choose("Furnished Status", [
            "Furnished", "Semi-Furnished", "Unfurnished"
        ])
        floor_no = st.number_input("Floor Number", 0, 100, 1)
        total_floors = st.number_input("Total Floors", 1, 100, 5)
        schools = st.number_input("Nearby Schools", 0, 100, 3)
        hospitals = st.number_input("Nearby Hospitals", 0, 100, 2)
        transport = choose("Public Transport Accessibility", [
            "High", "Medium", "Low"
        ])
        parking = choose("Parking Space", ["Yes", "No"])
        security = choose("Security", ["Yes", "No"])

    with c3:
        amenities = choose("Amenities", ["Basic", "Moderate", "Premium"])
        facing = choose("Facing", ["North", "South", "East", "West"])
        owner_type = choose("Owner Type", ["Owner", "Broker", "Builder"])
        availability = choose("Availability Status", [
            "Ready_to_Move", "Under_Construction"
        ])

    submitted = st.form_submit_button("Analyze Property")

if submitted:
    if floor_no > total_floors:
        st.error("Floor Number cannot exceed Total Floors.")
        st.stop()

    age = max(0, 2026 - int(year_built))
    price_per_sqft = price * 100000 / size

    features = pd.DataFrame([{
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_per_SqFt": price_per_sqft,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability
    }])

    regression_features = features.copy()
    regression_features["Price_in_Lakhs"] = price

    try:
        prediction = classification_model.predict(features)[0]
        future_price = regression_model.predict(regression_features)[0]

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("Classification: Good Investment")
            else:
                st.warning("Classification: Not Good Investment")
            st.caption(
                "This model was trained using a median-price-based label. "
                "It does not establish actual profitability."
            )

        with col2:
            st.metric("Current Price", f"₹{price:,.2f} lakhs")
            st.metric(
                "Estimated Price in 5 Years",
                f"₹{future_price:,.2f} lakhs"
            )

        st.info(
            "The five-year target was created using an assumed 8% annual "
            "appreciation rate. It is a scenario estimate, not a validated "
            "prediction of actual future sale prices."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
