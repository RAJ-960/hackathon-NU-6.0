import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pygsheets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

def open_google_sheet(credentials_file, sheet_title, worksheet_title):
    try:
        gc = pygsheets.authorize(service_file=credentials_file)
        sh = gc.open(sheet_title)
        worksheet = sh.worksheet_by_title(worksheet_title)
        return worksheet
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def write_to_worksheet(worksheet, data):
    if isinstance(data, pd.DataFrame):
        existing_data = worksheet.get_as_df()
        updated_data = pd.concat([existing_data, data], ignore_index=True)
        worksheet.set_dataframe(updated_data, start='A1', index=False, header=True)

def load_and_clean_data(file):
    df = pd.read_excel(file, sheet_name="Sheet1")
    return df.dropna().reset_index(drop=True)

def extract_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    median_color = np.median(image.reshape(-1, 3), axis=0).astype(int)
    return median_color

def is_soil_color(color):
    color = np.array(color)
    lower_brown, upper_brown = np.array([60, 30, 15]), np.array([210, 160, 120])
    lower_black, upper_black = np.array([0, 0, 0]), np.array([70, 70, 70])
    return (np.all(color >= lower_brown) and np.all(color <= upper_brown)) or (np.all(color >= lower_black) and np.all(color <= upper_black))

def predict_soil_type(color):
    if is_soil_color(color):
        return "Black Soil" if np.all(color <= [50, 50, 50]) else "Brown Soil"
    return "Unknown Soil Type"

def check_soil_safety(moisture, temp, ph):
    safe, suggestions = True, []
    if moisture < 10:
        safe, suggestions.append("Increase water supply.")
    elif moisture > 40:
        safe, suggestions.append("Improve drainage.")
    if temp < 15:
        safe, suggestions.append("Ensure proper sunlight.")
    elif temp > 40:
        safe, suggestions.append("Provide shade and water.")
    if ph < 5.5:
        safe, suggestions.append("Add lime to reduce acidity.")
    elif ph > 7.5:
        safe, suggestions.append("Add organic matter to reduce alkalinity.")
    return safe, suggestions

def generate_bar_chart(moisture, temp, ph):
    fig, ax = plt.subplots(facecolor='#121212')
    ax.bar(['Moisture', 'Temperature', 'pH'], [moisture, temp, ph], color=['blue', 'red', 'green'])
    ax.set_xlabel("Parameters", color='white')
    ax.set_ylabel("Values", color='white')
    ax.set_title("Soil Parameters Bar Chart", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#121212')
    return fig

def analyze_soil(image, excel_file):
    df = load_and_clean_data(excel_file)
    dominant_color = extract_dominant_color(image)
    soil_type = predict_soil_type(dominant_color)
    if soil_type == "Unknown Soil Type":
        return {"Error": "Uploaded image is not a soil image."}, None
    moisture, temp, ph = df['Moist'].sample(1).iloc[0], df['Temp'].sample(1).iloc[0], df['Ph'].sample(1).iloc[0]
    safe, suggestions = check_soil_safety(moisture, temp, ph)
    graph = generate_bar_chart(moisture, temp, ph)
    result = {
        "Dominant Soil Color (RGB)": dominant_color.tolist(),
        "Predicted Soil Type": soil_type,
        "Moisture": round(moisture, 2),
        "Temperature": round(temp, 2),
        "pH Level": round(ph, 2),
        "Soil Safety": "Safe" if safe else "Unsafe",
        "Suggestions": suggestions
    }
    credentials_file = "soilautomate-99da64116b0b.json"
    sheet_title = "Soilautomate"
    worksheet_title = "Sheet1"
    worksheet = open_google_sheet(credentials_file, sheet_title, worksheet_title)
    if worksheet:
        df_to_write = pd.DataFrame([{**result}])
        write_to_worksheet(worksheet, df_to_write)
    return result, graph

st.set_page_config(page_title="Soil Analysis", layout="wide")

st.markdown(
    """
    <style>
    .big-font { font-size:30px !important; color: #4CAF50; }
    .stApp { background-color: #121212; color: white; }
    .sidebar .sidebar-content { background-color: #333333; color: white; }
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Soil Analysis", "About"], key="navigation")

if selection == "Home":
    st.title("🌱 Welcome to Soil Analysis System")
    st.write("This application helps analyze soil properties based on an image and data file.")
    st.balloons()

elif selection == "Soil Analysis":
    st.title("🔬 Soil Analysis Dashboard")
    image_file = st.file_uploader("Upload Soil Image", type=['jpg', 'png', 'jpeg'])
    excel_file = st.file_uploader("Upload Excel Data", type=['xlsx'])
    if image_file and excel_file:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        result, graph = analyze_soil(image, excel_file)
        st.json(result)
        st.pyplot(graph)
        st.success("Analysis Complete!")

elif selection == "About":
    st.title("ℹ️ About Soil Analysis System")
    st.write("This system predicts soil type and suggests improvements based on moisture, temperature, and pH levels.")
    st.snow()