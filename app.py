import streamlit as st
import pandas as pd
import joblib
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus
import json
from datetime import datetime

from utils.preprocess import load_tourism_data, feature_engineer_date
from utils.fetch_weather import get_weather_on_date
from utils.visualize import plot_historical_patterns

ROOT = Path(__file__).resolve().parent
HISTORY_FILE = ROOT / "data" / "trip_history.json"
st.set_page_config(page_title="Uttarakhand Travel Guide", page_icon="🏔️", layout="wide")
st.markdown("<style>.hero{padding:1.4rem;border-radius:16px;background:linear-gradient(120deg,#0b3d3a,#176b63);color:white}.hero h1{margin:0}</style><div class='hero'><h1>🏔️ Uttarakhand Travel Guide</h1><p>Know the rush, weather, hotels and places to visit before you go.</p></div>", unsafe_allow_html=True)

PLACES = {
 "Almora":["Bright End Corner","Kasar Devi Temple","Chitai Golu Devta Temple"], "Auli":["Auli Ropeway","Gurso Bugyal","Chenab Lake"],
 "Badrinath":["Badrinath Temple","Mana Village","Vasudhara Falls"], "Bhimtal":["Bhimtal Lake","Victoria Dam","Aquarium Island"],
 "Dehradun":["Robber's Cave","Sahastradhara","Forest Research Institute"], "Haridwar":["Har Ki Pauri","Mansa Devi Temple","Ganga Aarti"],
 "Jim Corbett National Park":["Dhikala Zone","Corbett Waterfall","Sitabani Wildlife Reserve"], "Mussoorie":["Mall Road","Kempty Falls","Gun Hill Point"],
 "Nainital":["Naini Lake","Naina Devi Temple","Snow View Point"], "Rishikesh":["Laxman Jhula","Triveni Ghat","River Rafting"],
 "Kedarnath":["Kedarnath Temple","Bhairavnath Temple","Vasuki Tal"], "Valley of Flowers":["Valley of Flowers","Hemkund Sahib","Govindghat"]
}

@st.cache_resource
def assets():
    # Must be the same file train_model.py built its features from. processed_data.csv
    # holds the already-aggregated hist_avg_* columns but not the raw temp/precip/
    # event_count/trend_score ones, so feature_engineer_date silently produced 0 for
    # four of the nine features the model expects.
    d = load_tourism_data(ROOT/"data"/"tourism_data.csv")
    return d, joblib.load(ROOT/"models"/"tourism_classifier.pkl"), joblib.load(ROOT/"models"/"site_encoder.pkl"), joblib.load(ROOT/"models"/"target_encoder.pkl")

df, model, site_encoder, target_encoder = assets()
sites = sorted(df.site.unique())

def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

def save_search(origin, site, trip_date, condition, visitors):
    history = load_history()
    history.insert(0, {"origin": origin, "site": site, "date": str(trip_date),
                       "condition": condition, "visitors": round(visitors),
                       "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history[:10], indent=2), encoding="utf-8")

with st.sidebar:
    st.header("Plan your trip")
    origin = st.text_input("Starting location", "New Delhi")
    trip_date = st.date_input("Travel date", date.today(), min_value=date.today())
    site = st.selectbox("Destination", sites)
    show_weather = st.checkbox("Load live weather", value=False)
    check = st.button("Show travel details", type="primary", width="stretch")
    history = load_history()
    if history:
        st.markdown("#### Recent searches")
        for item in history[:5]:
            st.caption(f"{item['site']} · {item['date']} · {item['condition']}")

if not check:
    st.info("Enter your starting location, choose a destination and date, then click **Show travel details**.")
    st.markdown("### Your trip summary will include")
    st.write("Visitor rush • Climate • Hotel price guidance • Famous places • Route and estimated travel time")
    st.stop()

dt = pd.Timestamp(trip_date)
feat = feature_engineer_date(df, site, dt)
feat["site"] = site_encoder.transform([site])[0]
feat = feat.reindex(columns=model.feature_names_in_, fill_value=0)
condition = target_encoder.inverse_transform([model.predict(feat)[0]])[0]
confidence = float(model.predict_proba(feat).max())
avg = float(feat.hist_avg_tourists.iloc[0])
rush = {"Mild":"Low rush", "Suitable":"Moderate rush", "Overcrowded":"High rush"}.get(condition, condition)
save_search(origin, site, trip_date, condition, avg)

st.subheader(f"{site} on {dt.strftime('%A, %d %B %Y')}")
a, b, c = st.columns(3)
a.metric("Visitor rush", rush)
b.metric("Expected visitors", f"{avg:,.0f}")
c.metric("Prediction confidence", f"{confidence:.0%}")
if condition == "Overcrowded":
    st.warning("This may be busy. Consider an early start and advance hotel booking.")
elif condition == "Mild":
    st.success("This is usually a quieter period for visitors.")
else:
    st.info("A balanced visitor level is expected.")

st.markdown("### Climate")
climate = {1:("Cool and dry",8,20),2:("Pleasant",10,23),3:("Mild",14,27),4:("Warm",18,31),5:("Warm",21,34),6:("Monsoon begins",20,30),7:("Rainy",19,28),8:("Rainy",18,27),9:("Fresh after monsoon",17,27),10:("Clear and cool",12,25),11:("Cool",8,21),12:("Cold",5,18)}[dt.month]
x, y, z = st.columns(3)
x.metric("Typical climate", climate[0])
y.metric("Typical low", f"{climate[1]}°C")
z.metric("Typical high", f"{climate[2]}°C")
if show_weather:
    w = get_weather_on_date(site, trip_date)
    if "error" in w:
        st.caption(w["error"])
    else:
        st.caption(f"Live forecast: {w['conditions']} · {w['temp']}°C · humidity {w['humidity']}%")

st.markdown("### Hotel price guide")
h1, h2, h3 = st.columns(3)
h1.metric("Budget stays", "₹800–₹2,000/night")
h2.metric("Comfort stays", "₹2,000–₹5,000/night")
h3.metric("Premium stays", "₹5,000+/night")
st.caption("Indicative ranges, not live quotes. Prices vary by season, weekend and availability.")

st.markdown("### Places to visit")
st.write("  •  ".join(PLACES.get(site, [site+" local sightseeing"])[:6]))

st.markdown("### Route from your location")
maps = f"https://www.google.com/maps/dir/?api=1&origin={quote_plus(origin)}&destination={quote_plus(site+', Uttarakhand')}&travelmode=driving"
embed = f"https://www.google.com/maps/dir/{quote_plus(origin)}/{quote_plus(site+', Uttarakhand')}?output=embed"
st.iframe(embed, height=450)
st.link_button(f"🗺️ Open full route: {origin} → {site}", maps)
st.caption("The embedded map shows the route in this page. Open the full route for live traffic and exact travel time.")

with st.expander("See historical visitor pattern"):
    st.pyplot(plot_historical_patterns(df, site, dt), width="stretch")
