import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pydeck as pdk
import jwt
import datetime
import hashlib
from pymongo import MongoClient
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------ App Config ------------------
st.set_page_config(page_title="Climate Dashboard Pro", layout="wide")
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["climate_app"]
users_col = db["users"]
audit_col = db["audit"]
SECRET_KEY = "MY_SECRET_CHANGE_THIS"

country_to_continent = {
    "Pakistan": "Asia", "India": "Asia", "China": "Asia", "Japan": "Asia",
    "United States": "North America", "United Kingdom": "Europe", "UK": "Europe",
    "Canada": "North America", "Mexico": "North America",
    "Brazil": "South America", "Argentina": "South America",
    "France": "Europe", "Germany": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Kenya": "Africa", "Egypt": "Africa", "South Africa": "Africa", "Nigeria": "Africa",
    "Australia": "Oceania", "New Zealand": "Oceania"
}

# ------------------ Password Hashing ------------------
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed

# ------------------ Initialize Default Users ------------------
if users_col.count_documents({}) == 0:
    users_col.insert_many([
        {"username": "admin", "password": hash_password("admin123"), "role": "admin"},
        {"username": "user", "password": hash_password("user123"), "role": "user"}
    ])

# ------------------ Logging & JWT ------------------
def log_action(user, action, details=None):
    audit_col.insert_one({
        "user": user,
        "action": action,
        "details": details,
        "ts": datetime.datetime.utcnow()
    })

def create_token(username, role):
    payload = {"username": username, "role": role, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        return None

# ------------------ Login ------------------
def login_ui():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user = users_col.find_one({"username": u})
        if user and verify_password(p, user["password"]):
            token = create_token(u, user["role"])
            st.session_state["token"] = token
            log_action(u, "login", {"success": True})
            st.experimental_set_query_params(refresh=datetime.datetime.utcnow().timestamp())
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
            log_action(u if u else "unknown", "login_failed")

# ------------------ Load & Clean Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/data.csv")
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df = df.dropna(subset=['dt'])
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['Continent'] = df['Country'].map(country_to_continent).fillna("Other")
    
    def clean_coord(val):
        if pd.isna(val): return np.nan
        val = str(val).strip().upper()
        if val.endswith('N'): return float(val.replace('N',''))
        elif val.endswith('S'): return -float(val.replace('S',''))
        elif val.endswith('E'): return float(val.replace('E',''))
        elif val.endswith('W'): return -float(val.replace('W',''))
        try: return float(val)
        except: return np.nan

    df['Latitude'] = df['Latitude'].apply(clean_coord)
    df['Longitude'] = df['Longitude'].apply(clean_coord)
    return df

# ------------------ Admin Management ------------------
def admin_user_management():
    st.header("Admin — User Management")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Create / Edit User")
        new_user = st.text_input("Username", key="new_user")
        new_pass = st.text_input("Password", type="password", key="new_pass")
        role = st.selectbox("Role", ["user","admin"], key="new_role")
        if st.button("Create / Update"):
            if new_user and new_pass:
                users_col.update_one({"username": new_user},
                                     {"$set":{"password": hash_password(new_pass), "role": role}}, upsert=True)
                st.success(f"User {new_user} created/updated")
                log_action(st.session_state.get("user","admin"), "create_update_user", {"username": new_user, "role": role})
    with col2:
        st.subheader("Existing Users")
        users = list(users_col.find({}, {"password":0}))
        dfu = pd.DataFrame(users)
        if not dfu.empty:
            st.dataframe(dfu)
            sel = st.selectbox("Select user to delete", dfu['username'].tolist())
            if st.button("Delete user"):
                users_col.delete_one({"username": sel})
                st.success(f"Deleted {sel}")
                log_action(st.session_state.get("user","admin"), "delete_user", {"username": sel})
        else:
            st.info("No users found")

# ------------------ Visualization ------------------
def show_dashboard(df):
    st.header("🌍 Climate Dashboard")

    # ---------------- Sidebar Filters ----------------
    st.sidebar.markdown("### Filters")
    years = st.sidebar.slider("Select Year Range", int(df.year.min()), int(df.year.max()), (1900, 2013))
    countries = st.sidebar.multiselect("Select Countries", sorted(df['Country'].dropna().unique()), default=None)
    sample_size = st.sidebar.slider("Sample size for map plots", 500, 5000, 2000)
    viz_choice = st.sidebar.selectbox("Select Visualization", [
        "Global Temperature Over Time","Monthly Pattern","Hottest Cities","Coldest Cities",
        "Hemisphere Comparison","Continent Trends","World Interactive Map","PyDeck Map"
    ])
    
    # ---------------- Apply Filters ----------------
    dff = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]
    if countries: 
        dff = dff[dff['Country'].isin(countries)]
    
    # ---------------- Visualization ----------------
    if viz_choice=="Global Temperature Over Time":
        t = dff.groupby("year")["AverageTemperature"].mean().dropna()
        fig,ax = plt.subplots()
        ax.plot(t.index, t.values)
        ax.set_title("Global Temperature Over Time")
        st.pyplot(fig)
    
    elif viz_choice=="Monthly Pattern":
        fig,ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x="month", y="AverageTemperature", data=dff, ax=ax)
        st.pyplot(fig)
    
    elif viz_choice=="Hottest Cities":
        t = dff.groupby("City")["AverageTemperature"].mean().sort_values(ascending=False).head(20)
        fig,ax = plt.subplots(figsize=(10,6))
        t.plot(kind="bar", ax=ax)
        st.pyplot(fig)
    
    elif viz_choice=="Coldest Cities":
        t = dff.groupby("City")["AverageTemperature"].mean().sort_values().head(20)
        fig,ax = plt.subplots(figsize=(10,6))
        t.plot(kind="bar", ax=ax)
        st.pyplot(fig)
    
    elif viz_choice=="Hemisphere Comparison":
        dff["hemisphere"] = dff["Latitude"].apply(lambda x: "North" if x>=0 else "South")
        t = dff.groupby(["hemisphere","year"])["AverageTemperature"].mean().reset_index()
        fig,ax = plt.subplots()
        for h in t['hemisphere'].unique():
            sub = t[t.hemisphere==h]
            ax.plot(sub.year, sub.AverageTemperature, label=h)
        ax.legend()
        st.pyplot(fig)
    
    elif viz_choice=="Continent Trends":
        t = dff.groupby(["Continent","year"])["AverageTemperature"].mean().reset_index()
        fig,ax = plt.subplots()
        for cont in t['Continent'].unique():
            sub = t[t.Continent==cont]
            ax.plot(sub.year, sub.AverageTemperature, label=cont)
        ax.legend()
        st.pyplot(fig)
    
    elif viz_choice=="World Interactive Map":
        samp = dff.dropna(subset=["Latitude","Longitude","AverageTemperature"]).sample(min(sample_size,len(dff)))
        fig = px.scatter_geo(samp, lat="Latitude", lon="Longitude", color="AverageTemperature",
                             hover_name="City", projection="natural earth", title="World Temperature")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_choice=="PyDeck Map":
        samp = dff.dropna(subset=["Latitude","Longitude","AverageTemperature"]).sample(min(sample_size,len(dff)))
        layer = pdk.Layer("ScatterplotLayer",
                          data=samp,
                          get_position='[Longitude, Latitude]',
                          get_fill_color='[255*(AverageTemperature - AverageTemperature.min())/(AverageTemperature.max()-AverageTemperature.min()+1e-6), 100, 150]',
                          get_radius=50000,
                          pickable=True)
        view_state = pdk.ViewState(latitude=samp["Latitude"].mean(), longitude=samp["Longitude"].mean(), zoom=1)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text":"{City}\n{AverageTemperature}"})
        st.pydeck_chart(r)


# ------------------ Main App ------------------
def main_app():
    token = st.session_state.get("token")
    decoded = decode_token(token) if token else None
    if not decoded:
        login_ui()
        return
    st.session_state["user"] = decoded["username"]
    st.session_state["role"] = decoded["role"]

    st.sidebar.markdown(f"**Logged in as:** {st.session_state['user']} — {st.session_state['role']}")
    if st.sidebar.button("Logout"):
        log_action(st.session_state.get("user"), "logout")
        del st.session_state["token"]
        st.experimental_set_query_params(refresh=datetime.datetime.utcnow().timestamp())
        st.experimental_rerun()

    df = load_data()
    if st.session_state.get("role")=="admin":
        page = st.sidebar.radio("Main Menu", ["Dashboard","Admin"])
    else:
        page = st.sidebar.radio("Main Menu", ["Dashboard"])
    
    if page=="Admin":
        admin_user_management()
    else:
        show_dashboard(df)

if __name__=="__main__":
    main_app()
