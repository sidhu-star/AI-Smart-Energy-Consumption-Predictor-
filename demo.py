import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Smart Energy Predictor", page_icon="⚡", layout="wide")

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 1000
    hour = rng.integers(0, 24, n)
    day = rng.integers(0, 7, n)
    month = rng.integers(1, 13, n)
    temperature = rng.uniform(18, 40, n)
    humidity = rng.uniform(30, 90, n)
    occupancy = rng.integers(0, 21, n)
    previous = rng.uniform(0.5, 8.0, n)
    peak = (((hour >= 7) & (hour <= 10)) | ((hour >= 18) & (hour <= 22))).astype(float)
    energy = 0.5 + 0.20*occupancy + 0.08*np.maximum(temperature-24, 0) + 0.35*previous + 0.35*peak + rng.normal(0,0.15,n)
    X = pd.DataFrame({"temperature":temperature,"humidity":humidity,"occupancy":occupancy,"previous":previous,"hour":hour,"day":day,"month":month})
    model = RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42)
    model.fit(X, np.maximum(energy, 0.2))
    return model

model = train_model()
st.title("⚡ AI Smart Energy Consumption Predictor")
st.caption("Interactive engineering demo • Random Forest • synthetic training data")

with st.sidebar:
    st.header("Prediction Inputs")
    temperature = st.slider("Temperature (°C)", 15.0, 45.0, 30.0, 0.5)
    humidity = st.slider("Humidity (%)", 20, 95, 55)
    occupancy = st.slider("Occupancy (people)", 0, 30, 6)
    previous = st.slider("Previous consumption (kWh)", 0.2, 10.0, 3.0, 0.1)
    hour = st.slider("Hour of day", 0, 23, 19)
    day = st.selectbox("Day", list(range(7)), format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
    month = st.slider("Month", 1, 12, 9)
    tariff = st.number_input("Tariff (₹/kWh)", 1.0, 20.0, 8.0, 0.5)

row = pd.DataFrame([{"temperature":temperature,"humidity":humidity,"occupancy":occupancy,"previous":previous,"hour":hour,"day":day,"month":month}])
prediction = float(model.predict(row)[0])
level = "Low" if prediction < 2 else "Moderate" if prediction < 4 else "High"
daily = prediction * 24
cost = daily * tariff
peak_hour = max(range(24), key=lambda h: float(model.predict(pd.DataFrame([{"temperature":temperature,"humidity":humidity,"occupancy":occupancy,"previous":previous,"hour":h,"day":day,"month":month}]))[0]))

c1,c2,c3,c4 = st.columns(4)
c1.metric("Predicted / hour", f"{prediction:.2f} kWh")
c2.metric("Usage level", level)
c3.metric("Estimated 24h cost", f"₹{cost:,.0f}")
c4.metric("Predicted peak hour", f"{peak_hour}:00")

st.subheader("24-Hour Forecast")
hours = np.arange(24)
vals = [float(model.predict(pd.DataFrame([{"temperature":temperature,"humidity":humidity,"occupancy":occupancy,"previous":previous,"hour":int(h),"day":day,"month":month}]))[0]) for h in hours]
chart = pd.DataFrame({"Predicted kWh":vals}, index=hours)
st.line_chart(chart)

st.subheader("AI Energy-Saving Recommendations")
tips = []
if temperature > 28: tips.append("Higher temperature may increase cooling demand; optimize AC setpoints.")
if occupancy > 10: tips.append("High occupancy can increase load; schedule flexible equipment outside peak periods.")
if hour in list(range(7,11))+list(range(18,23)): tips.append("Current time is in a modeled peak window; consider shifting flexible loads.")
if not tips: tips.append("Current conditions indicate relatively moderate demand; continue monitoring.")
for t in tips: st.info("• " + t)

st.subheader("Input Summary")
st.dataframe(pd.DataFrame({"Parameter":["Temperature","Humidity","Occupancy","Previous usage","Hour","Day"],"Value":[f"{temperature} °C",f"{humidity} %",occupancy,f"{previous} kWh",f"{hour}:00",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][day]]}), hide_index=True, use_container_width=True)
st.divider()
st.caption("Educational demo: the model is trained on synthetic data. Replace the training generator with real smart-meter data for deployment.")
