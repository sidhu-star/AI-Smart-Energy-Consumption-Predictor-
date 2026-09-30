import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Smart Energy Predictor", page_icon="⚡", layout="wide")

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 2000
    hour = rng.integers(0, 24, n)
    day = rng.integers(0, 7, n)
    month = rng.integers(1, 13, n)
    temperature = rng.uniform(18, 40, n)
    humidity = rng.uniform(30, 90, n)
    occupancy = rng.integers(0, 21, n)
    previous = rng.uniform(0.5, 8.0, n)
    
    # More realistic energy model with multiple factors
    # Base load
    base_load = 1.0
    
    # Occupancy effect
    occupancy_effect = 0.25 * occupancy
    
    # Temperature effect (cooling/heating)
    temp_effect = 0.15 * np.abs(temperature - 24)
    
    # Time-of-day effect (stronger peaks)
    peak_morning = ((hour >= 7) & (hour <= 10)).astype(float) * 2.5
    peak_evening = ((hour >= 18) & (hour <= 22)).astype(float) * 3.0
    off_peak = ((hour >= 0) & (hour <= 6)).astype(float) * 0.6
    
    # Day-of-week effect (weekends lower)
    weekend_factor = ((day >= 5).astype(float) * 0.8) + ((day < 5).astype(float) * 1.0)
    
    # Seasonal effect
    summer_factor = (((month >= 5) & (month <= 9)).astype(float) * 1.2) + (((month < 5) | (month > 9)).astype(float) * 0.95)
    
    # Previous consumption momentum
    prev_momentum = 0.2 * previous
    
    # Combine all effects
    energy = base_load + occupancy_effect + temp_effect + peak_morning + peak_evening + off_peak + prev_momentum
    energy = energy * weekend_factor * summer_factor
    energy = np.maximum(energy, 0.3) + rng.normal(0, 0.2, n)
    energy = np.maximum(energy, 0.2)
    
    X = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "previous": previous,
        "hour": hour,
        "day": day,
        "month": month
    })
    model = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, min_samples_split=5)
    model.fit(X, energy)
    return model

model = train_model()
st.title("⚡ AI Smart Energy Consumption Predictor")
st.caption("Interactive engineering demo • Random Forest with realistic patterns • Occupancy, temperature, time-of-day, seasonal effects")

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

row = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "occupancy": occupancy,
    "previous": previous,
    "hour": hour,
    "day": day,
    "month": month
}])
prediction = float(model.predict(row)[0])
level = "Low" if prediction < 2 else "Moderate" if prediction < 4 else "High"
daily = prediction * 24
cost = daily * tariff
peak_hour = max(range(24), key=lambda h: float(model.predict(pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "occupancy": occupancy,
    "previous": previous,
    "hour": h,
    "day": day,
    "month": month
}]))[0]))

c1,c2,c3,c4 = st.columns(4)
c1.metric("Predicted / hour", f"{prediction:.2f} kWh")
c2.metric("Usage level", level)
c3.metric("Estimated 24h cost", f"₹{cost:,.0f}")
c4.metric("Predicted peak hour", f"{peak_hour}:00")

st.subheader("24-Hour Forecast")
hours = np.arange(24)
vals = [float(model.predict(pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "occupancy": occupancy,
    "previous": previous,
    "hour": int(h),
    "day": day,
    "month": month
}]))[0]) for h in hours]

chart_df = pd.DataFrame({
    "Hour": [f"{h:02d}:00" for h in hours],
    "Predicted kWh": vals
})
st.line_chart(chart_df.set_index("Hour"), use_container_width=True)

# Add pattern insights
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Peak consumption", f"{max(vals):.2f} kWh", f"at {hours[np.argmax(vals)]:02d}:00")
with col2:
    st.metric("Min consumption", f"{min(vals):.2f} kWh", f"at {hours[np.argmin(vals)]:02d}:00")
with col3:
    st.metric("Avg 24h", f"{np.mean(vals):.2f} kWh", f"Total: {np.sum(vals):.2f} kWh")

st.subheader("AI Energy-Saving Recommendations")
tips = []
if temperature > 28: tips.append("Higher temperature may increase cooling demand; optimize AC setpoints.")
if occupancy > 10: tips.append("High occupancy can increase load; schedule flexible equipment outside peak periods.")
if hour in list(range(7,11))+list(range(18,23)): tips.append("Current time is in a peak consumption window; consider shifting flexible loads.")
if temperature < 20: tips.append("Low temperature detected; ensure heating systems are efficient.")
if humidity > 70: tips.append("High humidity may increase cooling load; check ventilation systems.")
if not tips: tips.append("Current conditions indicate moderate demand; continue monitoring.")
for t in tips: st.info("• " + t)

st.subheader("Input Summary")
st.dataframe(pd.DataFrame({
    "Parameter": ["Temperature","Humidity","Occupancy","Previous usage","Hour","Day","Month"],
    "Value": [
        f"{temperature} °C",
        f"{humidity} %",
        occupancy,
        f"{previous} kWh",
        f"{hour}:00",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][day],
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month-1]
    ]
}), hide_index=True, use_container_width=True)

st.divider()
st.caption("Enhanced demo: includes realistic patterns for occupancy, temperature, time-of-day, day-of-week, and seasonal effects.")

