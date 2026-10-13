# 🚛 TradeFlow-AI: Enterprise Dynamic Pricing Engine

## 📌 Overview
TradeFlow-AI is an enterprise-grade Machine Learning pipeline designed to dynamically calculate logistics and freight spot prices. Built with XGBoost and Scikit-Learn, this pricing engine ingests real-time supply chain metrics, geographic data, and economic indicators to generate optimized, market-reflective freight rates.

## ✨ Key Features
* Real-Time Supply & Demand Economics: Dynamically adjusts prices based on the exact ratio of pending_loads to available_trucks.
* Geospatial Intelligence: Calculates exact haulage distances using the Haversine formula, adjusted for local road sinuosity.
* Hierarchical Urgency Processing: Utilizes OrdinalEncoder to understand the exact business weight of cargo urgency (`Standard` -> Express -> `Critical`).
* Multi-Factor Risk & Economy Adjustments: Factors in weather conditions, road quality, fuel shortages, and forex pressure indexes.
* Robust ML Pipeline: Fully unified scikit-learn Pipeline containing both the ColumnTransformer (preprocessing) and the XGBRegressor, exported as a single .pkl file for seamless production deployment.

## 📂 Repository Structure
* tradeflow_Dynamic_pricing_dataset.csv: The synthetic enterprise dataset containing 250,000+ generated logistics records.
* dynamic_pricing.ipynb: The complete Google Colab notebook containing data generation, feature engineering, model training, and diagnostic evaluations.
* tradeflow_Dynamic_pricing_engine.pkl: The compiled and serialized Machine Learning pipeline ready for API integration (e.g., FastAPI/Flask).

## 🚀 Getting Started (Production Deployment)

To use the dynamic pricing engine in your backend application, simply load the unified pipeline using joblib. The pipeline automatically handles all feature scaling (StandardScaler) and categorical encoding (OneHot/Ordinal).

```python
import joblib
import pandas as pd

# 1. Load the pre-trained Enterprise Pipeline
pricing_engine = joblib.load('tradeflow_Dynamic_pricing_engine.pkl')

# 2. Example Incoming Request (Real-time Fast API payload)
new_freight_request = pd.DataFrame([{
    'distance_km': 551.91,
    'cargo_weight_tons': 40.5,
    'fuel_price_index': 105.2,
    'available_trucks': 12,
    'pending_loads': 45,
    'supply_demand_ratio': 3.75, # Calculated in backend (45/12)
    'road_condition_factor': 1.15,
    'fuel_shortage_level': 1,
    'weather_condition': 0,
    'customs_dwell_time_hours': 24,
    'backhaul_factor': 1,
    'cargo_risk_tier': 2,
    'urgency_level': 'Express', # Model natively understands text due to OrdinalEncoder
    'seasonality_month': 8,
    'route_risk_index': 0.45,
    'vehicle_type': 2,
    'forex_pressure_index': 1.5,
    'corridor_tolls_etb': 4500.00
}])

# 3. Predict the exact spot price
predicted_price = pricing_engine.predict(new_freight_request)
print(f"Optimal Spot Price: {predicted_price[0]:.2f} ETB")
