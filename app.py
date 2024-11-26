import streamlit as st
import pandas as pd
import requests
import pickle
import sqlite3
from PIL import Image
import bcrypt
import hashlib



# API credentials
API_KEY = "661e31209c95328976a7cdc51aebf03f"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


# Load models
with open('best_rfc.pkl', 'rb') as crop_model_file:
    crop_model = pickle.load(crop_model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('crop_nutrient_model.pkl', 'rb') as nutrient_model_file:
    nutrient_model = pickle.load(nutrient_model_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)



# Crop dictionary
crop_dict = {
    0: 'Rice', 1: 'Maize', 2: 'Jute', 3: 'Cotton', 4: 'Coconut', 5: 'Papaya', 6: 'Orange', 7: 'Apple',
    8: 'Muskmelon', 9: 'Watermelon', 10: 'Grapes', 11: 'Mango', 12: 'Banana', 13: 'Pomegranate',
    14: 'Lentil', 15: 'Blackgram', 16: 'MungBean', 17: 'MothBeans', 18: 'PigeonPeas', 19: 'KidneyBeans',
    20: 'ChickPea', 21: 'Coffee'
}

# Reverse mapping for nutrient prediction
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# Function to fetch 5-day weather forecast
def fetch_weather_data(city_name):
    try:
        params = {"q": city_name, "appid": API_KEY, "units": "metric"}
        response = requests.get(FORECAST_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        temp = [forecast['main']['temp'] for forecast in data['list']]
        humidity = [forecast['main']['humidity'] for forecast in data['list']]
        avg_temp = sum(temp) / len(temp)
        avg_humidity = sum(humidity) / len(humidity)
        return avg_temp, avg_humidity
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None, None

    


#Database setup
DB_FILE = "users.db"
FEEDBACK_TABLE = "feedback"

# Initialize the SQLite database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")


# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User authentication functions
def signup_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    return cursor.fetchone() is not None

# App begins
st.set_page_config(page_title="Crop & Nutrient Recommendation", layout="wide")

# Background style
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.transparenttextures.com/patterns/fancy-deboss.png");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# User session management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# Login/Signup system
if not st.session_state.logged_in:
    with st.sidebar:
        st.header("Authentication")
        choice = st.radio("Select Action", ["Login", "Signup"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Submit"):
            if choice == "Login":
                if login_user(username, password):
                    st.success("Logged in successfully!")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                else:
                    st.error("Invalid username or password.")
            elif choice == "Signup":
                if signup_user(username, password):
                    st.success("Account created successfully!")
                else:
                    st.error("Username already exists.")


# else:
#     with st.sidebar:
#         st.header(f"Welcome, {st.session_state.username}!")
#         if st.button("Logout"):
#             st.session_state.logged_in = False
#             st.session_state.username = None
#             st.experimental_rerun()







# Apply custom CSS for a transparent background image
st.markdown("""
    <style>
        body {
            background-image: url("https://i.imgur.com/ZvX3Xs6.png"); /* Replace with your image URL */
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }
        .transparent {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)
if st.session_state.logged_in:
    # Tabs for app navigation
    homepage, tab1, tab2,feedback_tab = st.tabs(["🏠 Home", "📋 Predict Crop", "🧪 Predict Nutrients","💬feedback"])
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.username}!")
        if st.button("logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()
    # Home Tab
    with homepage :
        st.markdown("""
                <div class="transparent">
                <h1 style="text-align: center;">🌾 Welcome to the Crop & Nutrient Recommendation System 🌾</h1>
                <p style="text-align: center; font-size: 18px;">
                This platform provides recommendations for the best crops based on soil nutrients and weather conditions.
                Additionally, it predicts the optimal nutrient requirements for selected crops to boost agricultural yields.
                </p>
                <div style="text-align: center;">
                <a href="https://github.com/your_github_username" target="_blank" style="margin: 0 10px;">🌐 GitHub</a>
            </div>
        </div>
            """, unsafe_allow_html=True)

    # Tab 1: Predict Crop
    with tab1:
        st.header("1️⃣ Soil Nutrient Details")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, step=1)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200, step=1)
        potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, step=1)
        ph_value = st.slider("pH Value", min_value=0.0, max_value=14.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)
        st.header("2️⃣ Weather Details")
        weather_input_method = st.radio("How would you like to provide weather details?",
                                     options=["Enter manually", "Fetch using city name"])
        if weather_input_method == "Enter manually":
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
        else:
            city_name = st.text_input("Enter City Name:")
            if city_name:
                avg_temp, avg_humidity= fetch_weather_data(city_name)
                if avg_temp is not None:
                    st.write(f"🌡️ Average Temperature (°C): {avg_temp:.2f}")
                    st.write(f"💧 Average Humidity (%): {avg_humidity:.2f}")
                    temperature = avg_temp
                    humidity = avg_humidity
                else:
                    st.warning("Unable to fetch weather data. Please try again.")

        if st.button("Predict Crop"):
            if temperature is not None and humidity is not None and rainfall is not None:
                input_data = scaler.transform([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
                prediction = crop_model.predict(input_data)
                predicted_crop = crop_dict.get(prediction[0], "Unknown Crop")
                st.success(f"🌱 The recommended crop for your field is: **{predicted_crop}**")
            else:
                st.warning("Please provide all the required inputs.")

    # Tab 2: Predict Nutrients
    with tab2:
        st.header("🌾 Select Crop to Predict Nutrient Requirements")
        crop_name = st.selectbox("Select a Crop", options=list(reverse_crop_dict.keys()))
        if st.button("Predict Nutrients"):
            crop_index = reverse_crop_dict[crop_name]
            encoded_crop = encoder.transform([[crop_name]])
            nutrient_prediction = nutrient_model.predict(encoded_crop)
            nutrients = nutrient_prediction[0]
            st.write("🧪 Predicted Nutrient Requirements and Conditions:")
            st.write(f"**Nitrogen (N):** {nutrients[0]:.2f}")
            st.write(f"**Phosphorus (P):** {nutrients[1]:.2f}")
            st.write(f"**Potassium (K):** {nutrients[2]:.2f}")
            st.write(f"**Temperature (°C):** {nutrients[3]:.2f}")
            st.write(f"**Humidity (%):** {nutrients[4]:.2f}")
            st.write(f"**pH Value:** {nutrients[5]:.2f}")
            st.write(f"**Rainfall (mm):** {nutrients[6]:.2f}")


    # Add the feedback CSV file path
    FEEDBACK_FILE = "feedback.csv"

    # Initialize the feedback storage file if it doesn't exist
    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame(columns=["Name", "Email", "Rating", "Comments"]).to_csv(FEEDBACK_FILE, index=False)

    # Feedback Tab
    with feedback_tab:
        st.header("💬 We Value Your Feedback!")

        # Input fields for feedback
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        rating = st.slider("Rate Your Experience (1 = Poor, 5 = Excellent)", 1, 5, 3)
        comments = st.text_area("Your Comments")

        # Submit feedback
        if st.button("Submit Feedback"):
            if name and email and comments:
                # Append feedback to the CSV file
                new_feedback = pd.DataFrame(
                    {"Name": [name], "Email": [email], "Rating": [rating], "Comments": [comments]}
                )
                new_feedback.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
                st.success("Thank you for your feedback! 💖")
            else:
                st.error("Please fill in all fields before submitting your feedback.")

        # Display submitted feedback
        if st.checkbox("View Feedback"):
            feedback_data = pd.read_csv(FEEDBACK_FILE)
            st.write("📋 Submitted Feedback:")
            st.dataframe(feedback_data)