import openrouteservice
import streamlit as st
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import folium
from streamlit_folium import folium_static
from transformers import pipeline
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Hugging Face API Key (Set your key in the environment variable or directly here)
HF_API_KEY = "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"
os.environ["HF_HOME"] = HF_API_KEY  # Set the Hugging Face API key

# ORS API Key
ORS_API_KEY = "5b3ce3597851110001cf6248a9522fa41b27439588c559d4621706db"
client = openrouteservice.Client(key=ORS_API_KEY)


# Hugging Face model for text generation
hf_pipeline = pipeline("text-generation", model="gpt2")  # You can replace with any suitable model

# Load models
model_customer = load_model("/content/customer_lstm_model.h5")
model_retailer = load_model("/content/retailer_mlp_model.h5")
model_logistics = load_model("/content/logistics_mlp_model.h5")

# API Key for OpenWeather API
OPENWEATHER_API_KEY = "1bb571682b348fe67ff37524f894fa37"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data.get("main") and data.get("wind") and "coord" in data:
        return {
            "temp": data["main"].get("temp", 25),
            "wind": data["wind"].get("speed", 5),
            "rain": data.get("rain", {}).get("1h", 0.0),
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"]
        }
    else:
        return {"temp": 25, "wind": 5, "rain": 0.0, "lat": 0.0, "lon": 0.0}

def generate_insight(city, product, score):
    context = (
        f"A customer in {city} has ordered a {product}. "
        f"The estimated delivery risk score is {score}. "
        "Based on this, provide a recommendation to the customer about whether to proceed with the order, and explain why."
    )
    result = hf_pipeline(context, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7)
    return result[0]["generated_text"]


# Title
st.title("AI-Powered Supply Chain Risk Predictor")

# Tabs
tab1, tab2, tab3 = st.tabs(["Customer", "Retailer", "Logistics"])

# Tab 1: Customer Risk Prediction
with tab1:
    st.header("Customer Risk Prediction")

    customer_city = st.text_input("Enter your city")
    product = st.text_input("Enter the product you want to order")

    if customer_city and product:
        if st.button("Check Risk and Get Recommendation"):
            with st.spinner("Fetching weather and analyzing risk..."):
                try:
                    weather = get_weather(customer_city)
                    st.write("### Weather Conditions at Your Location:")
                    st.json(weather)

                    # Show on map
                    m = folium.Map(location=[weather["lat"], weather["lon"]], zoom_start=10)
                    folium.Marker([weather["lat"], weather["lon"]], tooltip="Customer Location").add_to(m)
                    folium_static(m)

                    # Prepare input for model
                    input_array = np.array([[weather["temp"], weather["wind"], weather["rain"]]] * 7).reshape(1, 7, 3)
                    risk_score = int(model_customer.predict(input_array)[0][0] * 100)

                    st.metric("📊 Delivery Risk Score", risk_score)

                    # Generate AI-based insight
                    insight = generate_insight(customer_city, product, risk_score)
                    st.subheader("📝 AI Insight and Suggested Action")
                    st.info(insight)

                except Exception as e:
                    st.error(f"Error during risk assessment: {e}")

with tab2:
    st.header("Retailer Delivery Route Details")

    customer_city = st.text_input("Customer Location")
    warehouse_city = st.text_input("Warehouse Location", key="warehouse_location_retailer")

    if customer_city and warehouse_city:
        warehouse_weather = get_weather(warehouse_city)
        customer_weather = get_weather(customer_city)

        # Routing using ORS (OpenRouteService)
        start_coords = (warehouse_weather["lon"], warehouse_weather["lat"])
        end_coords = (customer_weather["lon"], customer_weather["lat"])

        try:
            route = client.directions(
                coordinates=[start_coords, end_coords],
                profile='driving-car',
                format='geojson',
                instructions=True
            )

            # Visualize on map
            m = folium.Map(location=[(warehouse_weather["lat"] + customer_weather["lat"]) / 2,
                                     (warehouse_weather["lon"] + customer_weather["lon"]) / 2], zoom_start=6)
            folium.Marker(
                [warehouse_weather["lat"], warehouse_weather["lon"]],
                tooltip="Warehouse", icon=folium.Icon(color="blue")
            ).add_to(m)

            folium.Marker(
                [customer_weather["lat"], customer_weather["lon"]],
                tooltip="Customer", icon=folium.Icon(color="green")
            ).add_to(m)

            # Draw route polyline
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in route['features'][0]['geometry']['coordinates']],
                color='purple', weight=4
            ).add_to(m)

            folium_static(m)

            # Extract route details
            properties = route["features"][0]["properties"]
            summary = properties["summary"]
            steps = properties.get("segments", [])[0].get("steps", [])

            route_details = {
                "from": warehouse_city,
                "to": customer_city,
                "distance_km": round(summary["distance"] / 1000, 2),
                "duration_minutes": round(summary["duration"] / 60, 2),
                "steps": [
                    {
                        "instruction": step["instruction"],
                        "distance_m": step["distance"],
                        "duration_s": step["duration"],
                        "type": step.get("type")
                    }
                    for step in steps
                ]
            }

            st.subheader("Route Summary")
            st.write(f"**Distance:** {route_details['distance_km']} km")
            st.write(f"**Estimated Time:** {route_details['duration_minutes']} minutes")

            st.subheader("Route Details")
            st.json(route_details)

        except Exception as e:
            st.error(f"Route generation error: {e}")

with tab3:
    st.header("Logistics Warehouse Route Details")

    city = st.text_input("Destination City")
    product = st.text_input("Product to Deliver")
    warehouse_city = st.text_input("Warehouse Location", key="warehouse_location_logistics")

    if city and warehouse_city and product:
        destination_weather = get_weather(city)
        warehouse_weather = get_weather(warehouse_city)

        start_coords = (warehouse_weather["lon"], warehouse_weather["lat"])
        end_coords = (destination_weather["lon"], destination_weather["lat"])

        try:
            route = client.directions(
                coordinates=[start_coords, end_coords],
                profile='driving-car',
                format='geojson',
                instructions=True
            )

            # Display map
            m = folium.Map(location=[
                (warehouse_weather["lat"] + destination_weather["lat"]) / 2,
                (warehouse_weather["lon"] + destination_weather["lon"]) / 2
            ], zoom_start=6)

            folium.Marker(
                [warehouse_weather["lat"], warehouse_weather["lon"]],
                tooltip="Warehouse", icon=folium.Icon(color="red")
            ).add_to(m)

            folium.Marker(
                [destination_weather["lat"], destination_weather["lon"]],
                tooltip="Destination", icon=folium.Icon(color="orange")
            ).add_to(m)

            folium.PolyLine(
                locations=[(c[1], c[0]) for c in route['features'][0]['geometry']['coordinates']],
                color='darkred', weight=4
            ).add_to(m)

            folium_static(m)

            # Extract route info
            properties = route["features"][0]["properties"]
            summary = properties["summary"]
            steps = properties.get("segments", [])[0].get("steps", [])

            route_details = {
                "from": warehouse_city,
                "to": city,
                "product": product,
                "distance_km": round(summary["distance"] / 1000, 2),
                "duration_minutes": round(summary["duration"] / 60, 2),
                "steps": [
                    {
                        "instruction": step["instruction"],
                        "distance_m": step["distance"],
                        "duration_s": step["duration"],
                        "type": step.get("type")
                    }
                    for step in steps
                ]
            }

            st.subheader("Logistics Route Summary")
            st.write(f"**Distance:** {route_details['distance_km']} km")
            st.write(f"**Estimated Time:** {route_details['duration_minutes']} minutes")

            st.subheader("Route Details")
            st.json(route_details)

        except Exception as e:
            st.error(f"Routing error: {e}")

