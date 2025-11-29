import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import jwt
import datetime
import hashlib
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

st.set_page_config(page_title="Climate Dashboard Pro", layout="wide", page_icon="🌐")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    * { margin: 0; padding: 0; }
    body { background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%); }
    
    .icon-text { display: flex; align-items: center; gap: 8px; }
    .section-header { 
        color: #00d4ff; 
        font-weight: 700; 
        margin-bottom: 15px; 
        font-size: 18px;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 8px;
    }
    .modern-btn { display: inline-flex; align-items: center; gap: 6px; }
    .login-label { color: #00d4ff; font-weight: 600; margin-bottom: 4px; margin-top: 0px; font-size: 14px; }
    .login-input { margin-bottom: 15px; }
    
    /* Dark theme background */
    [data-testid="stMainBlockContainer"] { background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%); }
    
    /* Button styling */
    button { 
        background: linear-gradient(135deg, #1f77b4 0%, #0f3460 100%) !important;
        color: #ffffff !important;
        border: 1px solid #00d4ff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    button:hover { 
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Slider styling - Blue theme */
    input[type="range"] { 
        accent-color: #00d4ff !important;
        -webkit-appearance: slider-horizontal;
    }
    input[type="range"]::-webkit-slider-thumb { 
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        background-color: #00d4ff !important;
        border: 2px solid #00d4ff !important;
        border-radius: 50% !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.8) !important;
        cursor: pointer !important;
    }
    input[type="range"]::-webkit-slider-runnable-track {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f2340 100%) !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    input[type="range"]::-moz-range-thumb { 
        appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        background-color: #00d4ff !important;
        border: 2px solid #00d4ff !important;
        border-radius: 50% !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.8) !important;
        cursor: pointer !important;
    }
    input[type="range"]::-moz-range-track { 
        background: transparent !important;
        border: none !important;
    }
    input[type="range"]::-moz-range-progress {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f2340 100%) !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    /* Streamlit slider component override - AGGRESSIVE */
    [data-testid="stSlider"] .stSlider {
        --slider-color: #00d4ff !important;
    }
    [data-testid="stSlider"] input[type="range"] {
        accent-color: #00d4ff !important;
        filter: hue-rotate(0deg) saturate(1) !important;
    }
    /* Override Streamlit's inline styles */
    .stSlider > div > div {
        --slider-color: #00d4ff !important;
    }
    /* Force color on slider thumb for Streamlit */
    [data-testid="stSlider"] {
        --slider-thumb-color: #00d4ff !important;
    }
    /* Safari specific */
    input[type="range"] {
        -webkit-appearance: slider-horizontal !important;
    }
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none !important;
    }
    
    /* Password input styling - hide default visibility toggle */
    input[type="password"] {
        font-size: 16px !important;
    }
    
    /* Hide password reveal button in Edge/IE */
    input::-ms-reveal, input::-ms-clear {
        display: none !important;
    }
</style>
<script>
    // Force slider color to blue
    function updateSliderColors() {
        const sliders = document.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            slider.style.setProperty('accent-color', '#00d4ff', 'important');
            slider.style.accentColor = '#00d4ff';
        });
    }
    updateSliderColors();
    window.addEventListener('load', updateSliderColors);
    const observer = new MutationObserver(updateSliderColors);
    observer.observe(document.body, { childList: true, subtree: true });
</script>
<style>
    
    /* Input fields */
    input, select, textarea { 
        background-color: #1a2332 !important;
        color: #ffffff !important;
        border: 1px solid #00d4ff !important;
        border-radius: 6px !important;
    }
    input::placeholder { color: #666 !important; }
    
    /* Success and error messages */
    .stSuccess { background-color: #1a5c3a !important; color: #00ff88 !important; border: 1px solid #00ff88 !important; }
    .stError { background-color: #5c1a1a !important; color: #ff6b6b !important; border: 1px solid #ff6b6b !important; }
    .stInfo { background-color: #1a3a5c !important; color: #00d4ff !important; border: 1px solid #00d4ff !important; }
    .stWarning { background-color: #5c4a1a !important; color: #ffb84d !important; border: 1px solid #ffb84d !important; }
    
    /* Text colors */
    h1, h2, h3 { color: #00d4ff !important; }
    p, span, label { color: #e0e0e0 !important; }
    
    /* Divider */
    hr { border-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

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

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed

def log_action(user, action, details=None):
    try:
        audit_col.insert_one({
            "user": user,
            "action": action,
            "details": details,
            "ts": datetime.datetime.utcnow()
        })
    except Exception:
        pass

def create_token(username, role):
    payload = {"username": username, "role": role, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token):
    if not token:
        return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except Exception:
        return None

def format_temperature(temp):
    if pd.isna(temp):
        return "N/A"
    return f"{temp:.1f}°C"

def describe_temperature(temp):
    if pd.isna(temp):
        return "unknown temperature"
    if temp < 0:
        return f"freezing ({format_temperature(temp)})"
    elif temp < 10:
        return f"very cold ({format_temperature(temp)})"
    elif temp < 15:
        return f"cold ({format_temperature(temp)})"
    elif temp < 20:
        return f"cool ({format_temperature(temp)})"
    elif temp < 25:
        return f"comfortable ({format_temperature(temp)})"
    elif temp < 30:
        return f"warm ({format_temperature(temp)})"
    else:
        return f"very hot ({format_temperature(temp)})"

def format_metric(label, value, unit=""):
    if isinstance(value, float):
        if value > 1:
            return f"{value:.2f}{unit}"
        else:
            return f"{value:.4f}{unit}"
    return f"{value}{unit}"

try:
    if users_col.count_documents({}) == 0:
        users_col.insert_many([
            {"username": "admin", "password": hash_password("admin123"), "role": "admin"},
            {"username": "user", "password": hash_password("user123"), "role": "user"}
        ])
except Exception:
    pass

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

def admin_user_management():
    st.markdown("<h3 class='section-header'><i class='fas fa-sliders-h'></i> User Management</h3>", unsafe_allow_html=True)
    
    with st.expander("<i class='fas fa-user-plus'></i> Add or Update User", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 13px; margin-bottom: 5px;'><i class='fas fa-user'></i> Username</p>", unsafe_allow_html=True)
            new_user = st.text_input("", key="new_user", label_visibility="collapsed")
            st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 13px; margin-bottom: 5px; margin-top: 10px;'><i class='fas fa-key'></i> Password</p>", unsafe_allow_html=True)
            new_pass = st.text_input("", type="password", key="new_pass", label_visibility="collapsed")
        with col2:
            st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 13px; margin-bottom: 5px;'><i class='fas fa-shield-alt'></i> Role</p>", unsafe_allow_html=True)
            role = st.selectbox("", ["user","admin"], key="new_role", label_visibility="collapsed")
            if st.button("Create / Update", use_container_width=False):
                if new_user and new_pass:
                    try:
                        users_col.update_one({"username": new_user},
                                             {"$set":{"password": hash_password(new_pass), "role": role}}, upsert=True)
                        st.markdown(f"<div style='background-color: #1a5c3a; color: #00ff88; border: 1px solid #00ff88; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> User {new_user} created/updated</div>", unsafe_allow_html=True)
                        log_action(st.session_state.get("user","admin"), "create_update_user", {"username": new_user, "role": role})
                    except Exception:
                        st.markdown("<div style='background-color: #5c1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-exclamation-circle'></i> Could not save user (check DB).</div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("<h3 class='section-header'><i class='fas fa-users'></i> Existing Users</h3>", unsafe_allow_html=True)
    
    try:
        users = list(users_col.find({}, {"password":0}))
        dfu = pd.DataFrame(users)
    except Exception:
        dfu = pd.DataFrame([])
    
    if not dfu.empty:
        st.dataframe(dfu, use_container_width=True)
        st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 13px; margin-bottom: 5px; margin-top: 15px;'><i class='fas fa-trash-alt'></i> Select User to Remove</p>", unsafe_allow_html=True)
        sel = st.selectbox("", dfu['username'].tolist(), label_visibility="collapsed")
        if st.button("Delete User"):
            try:
                users_col.delete_one({"username": sel})
                st.markdown(f"<div style='background-color: #1a5c3a; color: #00ff88; border: 1px solid #00ff88; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> Deleted {sel}</div>", unsafe_allow_html=True)
                log_action(st.session_state.get("user","admin"), "delete_user", {"username": sel})
            except Exception:
                st.markdown("<div style='background-color: #5c1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-exclamation-circle'></i> Could not delete user (DB error).</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: #1a3a5c; color: #00d4ff; border: 1px solid #00d4ff; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-info-circle'></i> No users found or DB unreachable</div>", unsafe_allow_html=True)

# ----------------- Dashboard UI -----------------
def show_dashboard(df):
    st.markdown("<h2 class='section-header'><i class='fas fa-chart-line'></i> Climate Analytics Dashboard</h2>", unsafe_allow_html=True)

    st.sidebar.markdown("<h3 class='section-header'><i class='fas fa-filter'></i> Data Filters</h3>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<p style='color: #00d4ff; font-weight: 700; font-size: 13px; margin-bottom: 5px;'><i class='fas fa-calendar-alt' style='margin-right: 8px;'></i>SELECT YEAR RANGE</p>", unsafe_allow_html=True)
    years = st.sidebar.slider("", int(df.year.min()), int(df.year.max()), (int(df.year.min()), int(df.year.max())), label_visibility="collapsed")
    
    st.sidebar.markdown("<p style='color: #00d4ff; font-weight: 700; font-size: 13px; margin-bottom: 5px; margin-top: 15px;'><i class='fas fa-globe' style='margin-right: 8px;'></i>SELECT COUNTRIES</p>", unsafe_allow_html=True)
    countries = st.sidebar.multiselect("", sorted(df['Country'].dropna().unique()), label_visibility="collapsed")

    cities = []
    if countries:
        cities = np.sort(df[df['Country'].isin(countries)]['City'].dropna().unique())
    
    st.sidebar.markdown("<p style='color: #00d4ff; font-weight: 700; font-size: 13px; margin-bottom: 5px; margin-top: 15px;'><i class='fas fa-city' style='margin-right: 8px;'></i>SELECT CITY</p>", unsafe_allow_html=True)
    selected_city = st.sidebar.selectbox("", options=["All"] + list(cities), label_visibility="collapsed")

    st.sidebar.markdown("<p style='color: #00d4ff; font-weight: 700; font-size: 13px; margin-bottom: 5px; margin-top: 15px;'><i class='fas fa-map-location-dot' style='margin-right: 8px;'></i>MAP SAMPLE SIZE</p>", unsafe_allow_html=True)
    sample_size = st.sidebar.slider("", 500, 5000, 2000, label_visibility="collapsed")

    st.sidebar.markdown("<h3 class='section-header'><i class='fas fa-chart-pie'></i> Choose Visualization</h3>", unsafe_allow_html=True)
    viz_choice = st.sidebar.radio("", [
        "Temperature Trends Over Time","Seasonal Patterns","Hottest Cities","Coldest Cities",
        "Hemisphere Comparison","Continental Analysis","Interactive Global Map","3D Heat Map","Temperature Forecast"
    ])

    dff = df[(df['year'] >= years[0]) & (df['year'] <= years[1])].copy()
    if countries:
        dff = dff[dff['Country'].isin(countries)].copy()
    if selected_city != "All":
        dff = dff[dff['City'] == selected_city].copy()

    st.markdown("<h3 class='section-header'><i class='fas fa-chart-bar'></i> Results & Analysis</h3>", unsafe_allow_html=True)

    if viz_choice == "Temperature Forecast":
        st.markdown("<h3 class='section-header'><i class='fas fa-crystal-ball'></i> Predict Temperature Trends</h3>", unsafe_allow_html=True)
        with st.form("predict_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 12px; margin-bottom: 5px;'><i class='fas fa-brain'></i> Model</p>", unsafe_allow_html=True)
                model_type = st.selectbox("", ["Linear Trend", "Advanced Curve Fitting", "Classification Model"], label_visibility="collapsed")
            with col2:
                st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 12px; margin-bottom: 5px;'><i class='fas fa-chart-line'></i> Based On</p>", unsafe_allow_html=True)
                x_feature = st.selectbox("", ["year", "month"], label_visibility="collapsed")
            with col3:
                st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 12px; margin-bottom: 5px;'><i class='fas fa-thermometer-half'></i> Predicting</p>", unsafe_allow_html=True)
                y_target = st.selectbox("", ["AverageTemperature"], label_visibility="collapsed")
            
            col4, col5 = st.columns(2)
            with col4:
                st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 12px; margin-bottom: 5px;'><i class='fas fa-chart-bar'></i> Metrics</p>", unsafe_allow_html=True)
                show_metrics = st.checkbox("Show Performance Metrics", value=True)
            with col5:
                st.markdown("<p style='color: #00d4ff; font-weight: 600; font-size: 12px; margin-bottom: 5px;'><i class='fas fa-hourglass-end'></i> Forecast</p>", unsafe_allow_html=True)
                future_years = st.slider("", 0, 20, 5, key="future_slider", label_visibility="collapsed")
            
            submit_pred = st.form_submit_button("Generate Forecast", use_container_width=True)
            if submit_pred:
                train = dff[[x_feature, y_target]].dropna()
                if train.empty:
                    st.markdown("<div style='background-color: #5c4a1a; color: #ffb84d; border: 1px solid #ffb84d; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-exclamation-triangle'></i> No data available for prediction with selected filters.</div>", unsafe_allow_html=True)
                else:
                    X = train[[x_feature]].values
                    y = train[y_target].values
                    
                    if model_type == "Linear Trend":
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        
                        st.markdown("<div style='background-color: #1a5c3a; color: #00ff88; border: 1px solid #00ff88; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> Linear trend model trained successfully!</div>", unsafe_allow_html=True)
                        result_df = pd.DataFrame({
                            x_feature: train[x_feature].values, 
                            "Actual": y, 
                            "Predicted": y_pred,
                            "Error": np.abs(y - y_pred)
                        })
                        
                        if show_metrics:
                            r2 = r2_score(y, y_pred)
                            rmse = np.sqrt(mean_squared_error(y, y_pred))
                            mae = mean_absolute_error(y, y_pred)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy (R²)", f"{r2:.4f}", help="1.0 = Perfect, 0.0 = Poor")
                            with col2:
                                st.metric("Avg Deviation (RMSE)", f"{rmse:.2f}°C")
                            with col3:
                                st.metric("Mean Error (MAE)", f"{mae:.2f}°C")
                        
                        fig = px.line(result_df.sort_values(x_feature), x=x_feature, y=["Actual", "Predicted"],
                                      labels={'value': 'Temperature (°C)', 'variable': 'Type'},
                                      title="Actual vs Predicted Temperatures",
                                      markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if x_feature == "year" and future_years > 0:
                            max_year = int(train[x_feature].max())
                            future_x = np.array([[max_year + i] for i in range(1, future_years + 1)])
                            future_pred = model.predict(future_x)
                            future_df = pd.DataFrame({
                                "Year": [max_year + i for i in range(1, future_years + 1)],
                                "Predicted Temperature": future_pred,
                                "Formatted": [format_temperature(t) for t in future_pred]
                            })
                            
                            st.subheader(f"⏳ Forecast for Next {future_years} Years")
                            
                            trend_change = future_pred[-1] - future_pred[0]
                            trend_direction = "warming trend" if trend_change > 0.5 else "cooling trend" if trend_change < -0.5 else "stable temperatures"
                            st.markdown(f"<div style='background-color: #1a3a5c; color: #00d4ff; border: 1px solid #00d4ff; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-arrow-trend-up'></i> Expected {trend_direction} ({trend_change:+.2f}°C change)</div>", unsafe_allow_html=True)
                            
                            st.dataframe(future_df[["Year", "Formatted"]].rename(columns={"Formatted": "Predicted Temp"}), 
                                        use_container_width=True, hide_index=True)
                            
                            combined_df = pd.concat([
                                result_df[[x_feature, "Predicted"]].rename(columns={"Predicted": "Temperature"}).assign(Type="Historical"),
                                pd.DataFrame({
                                    x_feature: [max_year + i for i in range(1, future_years + 1)],
                                    "Temperature": future_pred,
                                    "Type": "Forecasted"
                                })
                            ], ignore_index=True)
                            
                            fig_future = px.line(combined_df, x=x_feature, y="Temperature", color="Type",
                                                title="Historical & Forecasted Temperatures",
                                                markers=True,
                                                labels={"Temperature": "Temperature (°C)"})
                            st.plotly_chart(fig_future, use_container_width=True)
                        
                        with st.expander("📋 View Full Data Table"):
                            display_df = result_df.copy()
                            display_df["Actual"] = display_df["Actual"].apply(format_temperature)
                            display_df["Predicted"] = display_df["Predicted"].apply(format_temperature)
                            display_df["Error"] = display_df["Error"].apply(lambda x: f"±{x:.2f}°C")
                            st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)
                    
                    elif model_type == "Advanced Curve Fitting":
                        degree = st.slider("Complexity Level", 2, 5, 2, key="poly_degree")
                        poly_features = PolynomialFeatures(degree=degree)
                        X_poly = poly_features.fit_transform(X)
                        
                        model = LinearRegression()
                        model.fit(X_poly, y)
                        y_pred = model.predict(X_poly)
                        
                        st.markdown(f"<div style='background-color: #1a5c3a; color: #00ff88; border: 1px solid #00ff88; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> Advanced model (level {degree}) trained successfully!</div>", unsafe_allow_html=True)
                        result_df = pd.DataFrame({
                            x_feature: train[x_feature].values, 
                            "Actual": y, 
                            "Predicted": y_pred,
                            "Error": np.abs(y - y_pred)
                        })
                        
                        if show_metrics:
                            r2 = r2_score(y, y_pred)
                            rmse = np.sqrt(mean_squared_error(y, y_pred))
                            mae = mean_absolute_error(y, y_pred)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy (R²)", f"{r2:.4f}", help="1.0 = Perfect, 0.0 = Poor")
                            with col2:
                                st.metric("Avg Deviation (RMSE)", f"{rmse:.2f}°C")
                            with col3:
                                st.metric("Mean Error (MAE)", f"{mae:.2f}°C")
                            
                            fit_quality = "excellent" if r2 > 0.8 else "good" if r2 > 0.6 else "moderate" if r2 > 0.4 else "poor"
                            st.markdown(f"<div style='background-color: #1a3a5c; color: #00d4ff; border: 1px solid #00d4ff; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-lightbulb'></i> Model quality is **{fit_quality}** with **±{mae:.1f}°C** typical error</div>", unsafe_allow_html=True)
                        
                        fig = px.line(result_df.sort_values(x_feature), x=x_feature, y=["Actual", "Predicted"],
                                      labels={'value': 'Temperature (°C)', 'variable': 'Type'},
                                      title=f"Curve Fitting (Level {degree}) Results",
                                      markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("<i class='fas fa-table'></i> View Full Data Table"):
                            display_df = result_df.copy()
                            display_df["Actual"] = display_df["Actual"].apply(format_temperature)
                            display_df["Predicted"] = display_df["Predicted"].apply(format_temperature)
                            display_df["Error"] = display_df["Error"].apply(lambda x: f"±{x:.2f}°C")
                            st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)
                    
                    else:
                        y_bin = (y > np.median(y)).astype(int)
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X, y_bin)
                        y_pred = model.predict(X)
                        y_pred_proba = model.predict_proba(X)
                        
                        st.markdown("<div style='background-color: #1a5c3a; color: #00ff88; border: 1px solid #00ff88; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> Classification model trained successfully!</div>", unsafe_allow_html=True)
                        res = pd.DataFrame({
                            x_feature: X.flatten(), 
                            "Above Median": y_bin, 
                            "Predicted": y_pred,
                            "Confidence": np.max(y_pred_proba, axis=1)
                        })
                        
                        if show_metrics:
                            accuracy = accuracy_score(y_bin, y_pred)
                            correct = (y_pred == y_bin).sum()
                            total = len(y_bin)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Classification Accuracy", f"{accuracy:.1%}")
                            with col2:
                                st.metric("Correct Predictions", f"{correct}/{total}", delta=f"{(1-accuracy):.1%} errors")
                            
                            st.markdown(f"<div style='background-color: #1a3a5c; color: #00d4ff; border: 1px solid #00d4ff; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-double'></i> Model correctly classifies **{accuracy:.1%}** of temperatures</div>", unsafe_allow_html=True)
                        
                        fig = px.scatter(res, x=x_feature, y="Confidence",
                                        color="Predicted",
                                        labels={"Predicted": "Classification", "Confidence": "Confidence"},
                                        title="Classification Confidence Analysis",
                                        color_discrete_map={0: "#3498db", 1: "#e74c3c"})
                        fig.update_layout(yaxis_title="Model Confidence Score (0-1)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("📋 View Full Data Table"):
                            display_df = res.copy()
                            display_df["Above Median"] = display_df["Above Median"].map({0: "Below Average", 1: "Above Average"})
                            display_df["Predicted"] = display_df["Predicted"].map({0: "Below Average", 1: "Above Average"})
                            display_df["Confidence"] = (display_df["Confidence"] * 100).apply(lambda x: f"{x:.1f}%")
                            st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)
    else:
        with st.spinner("Analyzing data..."):
            if viz_choice == "Temperature Trends Over Time":
                t = dff.groupby("year")["AverageTemperature"].mean().dropna()
                fig = px.line(x=t.index, y=t.values, labels={'x':'Year','y':'Average Temperature (°C)'}, 
                             title="<i class='fas fa-arrow-trend-up'></i> Global Temperature Trends Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                if len(t) > 1:
                    temp_change = t.iloc[-1] - t.iloc[0]
                    direction = "warmed by" if temp_change > 0 else "cooled by"
                    st.caption(f"<i class='fas fa-thermometer-half' style='color: #00d4ff;'></i> Climate has {direction} {abs(temp_change):.2f}°C over {int(t.index[-1] - t.index[0])} years", unsafe_allow_html=True)

            elif viz_choice == "Seasonal Patterns":
                monthly = dff.groupby("month")["AverageTemperature"].agg(['mean', 'std', 'count']).reset_index()
                monthly["month_name"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                
                fig = px.box(dff, x="month", y="AverageTemperature", 
                            title="<i class='fas fa-circle-notch'></i> How Temperatures Change Seasonally",
                            labels={'month': 'Month', 'AverageTemperature': 'Temperature (°C)'})
                st.plotly_chart(fig, use_container_width=True)
                
                warmest_month = monthly.loc[monthly["mean"].idxmax()]
                coldest_month = monthly.loc[monthly["mean"].idxmin()]
                st.caption(f"<i class='fas fa-sun' style='color: #ffa500;'></i> Warmest: {warmest_month['month_name']} ({warmest_month['mean']:.1f}°C) | "
                          f"❄️ Coldest: {coldest_month['month_name']} ({coldest_month['mean']:.1f}°C)", unsafe_allow_html=True)

            elif viz_choice == "Hottest Cities":
                t = dff.groupby("City")["AverageTemperature"].mean().sort_values(ascending=False).head(20)
                fig = px.bar(x=t.index, y=t.values, labels={'x':'City','y':'Average Temperature (°C)'}, 
                            title="<i class='fas fa-fire'></i> Top 20 Hottest Cities")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"<i class='fas fa-fire' style='color: #ff6b6b;'></i> Hottest: {t.index[0]} averages {t.values[0]:.1f}°C", unsafe_allow_html=True)

            elif viz_choice == "Coldest Cities":
                t = dff.groupby("City")["AverageTemperature"].mean().sort_values().head(20)
                fig = px.bar(x=t.index, y=t.values, labels={'x':'City','y':'Average Temperature (°C)'}, 
                            title="<i class='fas fa-snowflake'></i> Top 20 Coldest Cities")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"<i class='fas fa-snowflake' style='color: #00bfff;'></i> Coldest: {t.index[0]} averages {t.values[0]:.1f}°C", unsafe_allow_html=True)

            elif viz_choice == "Hemisphere Comparison":
                dff_temp = dff.dropna(subset=["Latitude"])
                dff_temp["hemisphere"] = dff_temp["Latitude"].apply(lambda x: "Northern" if x >= 0 else "Southern")
                t = dff_temp.groupby(["hemisphere","year"])["AverageTemperature"].mean().reset_index()
                fig = px.line(t, x="year", y="AverageTemperature", color="hemisphere", 
                             title="<i class='fas fa-globe'></i> Comparing Hemispheres",
                             labels={'AverageTemperature': 'Average Temperature (°C)', 'year': 'Year'},
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_choice == "Continental Analysis":
                t = dff.groupby(["Continent","year"])["AverageTemperature"].mean().reset_index()
                fig = px.line(t, x="year", y="AverageTemperature", color="Continent", 
                             title="<i class='fas fa-map'></i> Temperature by Continent",
                             labels={'AverageTemperature': 'Average Temperature (°C)', 'year': 'Year'},
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_choice == "Interactive Global Map":
                samp = dff.dropna(subset=["Latitude","Longitude","AverageTemperature"])
                if len(samp) > sample_size:
                    samp = samp.sample(sample_size, random_state=0)
                fig = px.scatter_geo(samp, lat="Latitude", lon="Longitude", color="AverageTemperature",
                                     hover_name="City", projection="natural earth", 
                                     title="<i class='fas fa-earth-globe'></i> Global Temperature Map",
                                     color_continuous_scale="RdYlBu_r",
                                     labels={'AverageTemperature': 'Temperature (°C)'})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"<i class='fas fa-map-pin' style='color: #00d4ff;'></i> Showing {len(samp)} cities (Red = Hot, Blue = Cold)", unsafe_allow_html=True)

            elif viz_choice == "3D Heat Map":
                samp = dff.dropna(subset=["Latitude","Longitude","AverageTemperature"]).copy()
                if samp.empty:
                    st.markdown("<div style='background-color: #1a3a5c; color: #00d4ff; border: 1px solid #00d4ff; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-info-circle'></i> No location data available for this visualization.</div>", unsafe_allow_html=True)
                else:
                    if len(samp) > sample_size:
                        samp = samp.sample(sample_size, random_state=0)
                    
                    temp_min = samp["AverageTemperature"].min()
                    temp_max = samp["AverageTemperature"].max()
                    temp_range = temp_max - temp_min if temp_max > temp_min else 1
                    
                    samp["color_red"] = ((samp["AverageTemperature"] - temp_min) / temp_range * 255).astype(int)
                    samp["color"] = samp.apply(lambda row: [row["color_red"], 100, 150, 200], axis=1)
                    
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=samp,
                        get_position="[Longitude, Latitude]",
                        get_fill_color="color",
                        get_radius=50000,
                        pickable=True
                    )
                    view_state = pdk.ViewState(
                        latitude=samp["Latitude"].mean(), 
                        longitude=samp["Longitude"].mean(), 
                        zoom=1
                    )
                    r = pdk.Deck(
                        layers=[layer], 
                        initial_view_state=view_state, 
                        tooltip={"text": "{City}\nTemp: {AverageTemperature}°C"}
                    )
                    
                    st.pydeck_chart(r)
                    st.caption(f"<i class='fas fa-circle' style='color: #ff6b6b;'></i> Temperature scale: {temp_min:.1f}°C (dark) to {temp_max:.1f}°C (bright red)", unsafe_allow_html=True)

# ----------------- Main App -----------------
def main_app():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    token = st.session_state.get("token")
    decoded = decode_token(token) if token else None

    if not decoded or not st.session_state.get("authenticated"):
        st.markdown("""
        <div style='text-align: center; margin: 60px 0;'>
            <h1 style='color: #00d4ff; margin-bottom: 10px;'>
                <i class='fas fa-globe' style='font-size: 40px;'></i><br>Climate Analytics Dashboard
            </h1>
            <p style='color: #a0a0a0; font-size: 16px;'>Sign in to access temperature data and analysis</p>
        </div>
        """, unsafe_allow_html=True)

        if "login_user" not in st.session_state:
            st.session_state["login_user"] = ""
        if "login_pass" not in st.session_state:
            st.session_state["login_pass"] = ""

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<p class='login-label'><i class='fas fa-user' style='color: #00d4ff;'></i> Username</p>", unsafe_allow_html=True)
            st.session_state["login_user"] = st.text_input("", value=st.session_state["login_user"], placeholder="Enter your username", key="login_input_user", label_visibility="collapsed")
            
            st.markdown("<p class='login-label'><i class='fas fa-lock' style='color: #00d4ff;'></i> Password</p>", unsafe_allow_html=True)
            st.session_state["login_pass"] = st.text_input("", type="password", value=st.session_state["login_pass"], placeholder="Enter your password", key="login_input_pass", label_visibility="collapsed")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            if st.button("Sign In", use_container_width=True, key="login_button"):
                try:
                    user = users_col.find_one({"username": st.session_state["login_user"]})
                except Exception:
                    user = None

                if user and verify_password(st.session_state["login_pass"], user["password"]):
                    st.session_state["token"] = create_token(st.session_state["login_user"], user["role"])
                    st.session_state["user"] = st.session_state["login_user"]
                    st.session_state["role"] = user["role"]
                    st.session_state["authenticated"] = True
                    st.session_state["page"] = "Dashboard"
                    log_action(st.session_state["user"], "login", {"success": True})
                    st.markdown("<div style='background-color: #1a5c3a; color: #00ff88; border: 1px solid #00ff88; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> Welcome! Loading your dashboard...</div>", unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown("<div style='background-color: #5c1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-exclamation-circle'></i> Invalid username or password</div>", unsafe_allow_html=True)
                    log_action(st.session_state.get("login_user","unknown"), "login_failed")
                    st.session_state["login_pass"] = ""
        return

    if "user" not in st.session_state:
        st.session_state["user"] = decoded.get("username")
    if "role" not in st.session_state:
        st.session_state["role"] = decoded.get("role")

    st.sidebar.markdown(f"""
    <div style='padding: 15px; background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%); border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #00d4ff;'>
        <div style='display: flex; align-items: center; margin-bottom: 8px;'>
            <i class='fas fa-user-circle' style='color: #00d4ff; margin-right: 10px; font-size: 24px;'></i>
            <span style='color: #ffffff; font-size: 16px; font-weight: 900;'>{st.session_state['user'].upper()}</span>
        </div>
        <div style='margin-left: 34px;'>
            <span style='color: #00d4ff; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;'>
                <i class='fas fa-badge-check' style='margin-right: 5px;'></i>{st.session_state['role'].title()}
            </span>
        </div>
        <div style='margin-left: 34px; margin-top: 6px; font-size: 11px; color: #00d4ff;'>
            <i class='fas fa-check-circle' style='margin-right: 4px;'></i>Active Session
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Sign Out", use_container_width=True):
        log_action(st.session_state.get("user","unknown"), "logout")
        for k in ["token", "user", "role", "login_user", "login_pass", "authenticated", "page"]:
            if k in st.session_state:
                del st.session_state[k]
        st.markdown("<div style='background-color: #1a3a5c; color: #00d4ff; border: 1px solid #00d4ff; padding: 10px; border-radius: 6px; margin: 10px 0;'><i class='fas fa-check-circle'></i> Successfully signed out. Returning to login...</div>", unsafe_allow_html=True)
        st.rerun()

    st.sidebar.markdown("<h3 class='section-header'><i class='fas fa-bars'></i> Menu</h3>", unsafe_allow_html=True)
    if st.session_state.get("role") == "admin":
        menu_choice = st.sidebar.radio("", ["Dashboard", "Settings"], index=0 if st.session_state.get("page","Dashboard")=="Dashboard" else 1)
        st.session_state["page"] = menu_choice
    else:
        st.session_state["page"] = "Dashboard"

    df = load_data()

    if st.session_state["page"] == "Settings":
        admin_user_management()
    else:
        show_dashboard(df)


if __name__ == "__main__":
    main_app()
