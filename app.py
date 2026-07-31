import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import time

from utils.preprocess import load_tourism_data, feature_engineer_date
from utils.fetch_weather import get_weather_on_date
from utils.fetch_events import get_articles_for_location
from utils.fetch_trends import get_trend_score
from utils.visualize import plot_historical_patterns

df = load_tourism_data('data/processed_data.csv')
model = joblib.load('models/tourism_classifier.pkl')
site_encoder = joblib.load('models/site_encoder.pkl')
target_encoder = joblib.load('models/target_encoder.pkl')

st.set_page_config(page_title="Uttarakhand Trip Condition Predictor", layout="wide")
st.title("Uttarakhand Trip Condition Predictor")

with st.sidebar:
    st.header("Your Trip Details")
    trip_date = st.date_input("Select trip date", min_value=datetime.today())
    site = st.selectbox("Choose a tourist site", df['site'].unique())

if st.button("Predict"):
    feat = feature_engineer_date(df, site, pd.to_datetime(trip_date))

    feat['site'] = site_encoder.transform([site])[0]

    feat = feat.reindex(columns=model.feature_names_in_, fill_value=0)

    pred_class_encoded = model.predict(feat)[0]
    pred_class = target_encoder.inverse_transform([pred_class_encoded])[0]
    pred_proba = model.predict_proba(feat).max()

    st.subheader(f"On **{trip_date.strftime('%B %d, %Y')}** at **{site}**:")
    st.markdown(f"### Condition: **{pred_class}**")
    st.write(f"Confidence: **{pred_proba*100:.1f}%**")

    st.pyplot(plot_historical_patterns(df, site, trip_date))

    trend_score = get_trend_score(site)
    st.info(f"Google Trends score: {trend_score}")

    
    weather = get_weather_on_date(site, trip_date)
    st.markdown(f"### Weather Info:")
    if "error" in weather:
        st.info(f"Error: {weather["error"]}")
    else:
        st.markdown(f""" **{site} on {weather['date']}**    
            - **Temperature:** {weather['temp']}°C  
            - **Conditions:** {weather['conditions']}  
            - **Description:** {weather['description']}  
            - **Humidity:** {weather['humidity']}%  
            - **Wind Speed:** {weather['windspeed']} km/h  
            """)


    st.markdown(f"### Events on that date:")
    all_events = {}

    articles = get_articles_for_location(site, datetime.today())
    if articles:
        all_events[site] = articles[:3]  
    time.sleep(1)  

    if not all_events:
        st.info("\nNo events or mentions found in news near this date for any spot.")
    else:
        for spot, articles in all_events.items():
            st.markdown(f"\n Location: {spot}")
            for article in articles:
                md = (
                    f"### {article['title']}\n"
                    f"> {article.get('description', 'No description.')}\n\n"
                    f"[ Read the full article]({article['url']})\n"
                    "---"
                )
                st.markdown(md)

    

    