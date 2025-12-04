# import streamlit as st
# import requests
# from io import BytesIO
# import json
# from PIL import Image
# import time # For simulating loading

# # ==========================================================
# # 1. Configuration and CSS Fix & Enhanced Styling
# # ==========================================================

# # Use the environment variable or default to a local Flask server URL.
# # IMPORTANT: Adjust this URL if your Flask backend is hosted elsewhere!
# API_BASE_URL = "http://localhost:5000"

# st.set_page_config(
#     page_title="Diabetic Retinopathy AI Diagnostic",
#     layout="wide", # Use wide layout for more horizontal space
#     initial_sidebar_state="expanded"
# )

# # Initialize session state for input mode control
# if 'input_mode' not in st.session_state:
#     st.session_state.input_mode = None

# # Enhanced Custom CSS for a professional, card-based look
# custom_css = """
# <style>
# /* --- Core Page & Layout Adjustments --- */
# /* FIX: Aggressively remove top padding for the entire app and main sections */
# #stApp {
#     padding-top: 0px !important; 
#     margin-top: 0px !important;
# }
# section.main {
#     padding-top: 0px !important;
#     margin-top: 0px !important;
# }

# /* Ensure the Streamlit header is also compact if it exists */
# .stApp > header {
#     height: 0px !important; /* Hide Streamlit's default header */
#     visibility: hidden;
# }

# /* General body styling */
# body {
#     font-family: 'Inter', sans-serif;
#     color: #333; /* Darker text for readability */
#     background-color: #f0f2f6; /* Light grey background */
# }

# /* TIGHTEN: Reduce default margins on all headers to minimize vertical space */
# h1, h2, h3, h4, h5, h6 {
#     margin-top: 0.5em; 
#     margin-bottom: 0.5em; 
#     color: #212121; /* Darker header color, will be overridden below */
# }

# /* Streamlit Title specific styling */
# .stTitle {
#     color: #1a73e8; /* Google Blue equivalent */
#     font-weight: 700;
#     margin-bottom: 0.2em; /* Reduce space below title */
# }

# /* UNIFIED HEADING COLOR: Make all primary headings blue */
# .stCard h2, .result-card h2, .input-card h2, 
# .result-card h3, .stSidebar h3 {
#     color: #1a73e8 !important; /* Force all relevant headings to blue */
# }


# /* Markdown styling (e.g., for subheader descriptions) */
# .stMarkdown {
#     color: #555;
# }

# /* --- Card Styling --- */
# .stCard, .result-card, .input-card {
#     background-color: #ffffff; /* White background for cards */
#     border-radius: 12px; /* More rounded corners */
#     box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); /* Stronger, softer shadow */
#     padding: 25px; /* More internal padding */
#     margin-bottom: 25px; /* Space between cards */
#     border: 1px solid #e0e0e0; /* Subtle border */
#     min-height: 250px; /* Ensure cards have some minimum height */
# }

# /* Specific styling for the input method card */
# .input-card {
#     padding-top: 15px;
#     padding-bottom: 15px;
# }

# /* Styling for the image display card */
# .image-display-card {
#     background-color: #ffffff;
#     border-radius: 12px;
#     box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
#     padding: 15px; /* Slightly less padding to contain image */
#     margin-bottom: 25px;
#     border: 1px solid #e0e0e0;
#     text-align: center; /* Center image caption */
#     min-height: 400px; /* Taller card for the image */
#     display: flex;
#     flex-direction: column;
#     justify-content: flex-start; /* Align content to the top */
#     align-items: center;
# }
# .image-display-card img {
#     border-radius: 8px; /* Rounded corners for the image itself */
#     max-width: 100%;
#     height: auto;
#     margin-top: 10px; /* Space below the header */
# }
# .image-display-card .stCaption {
#     color: #777;
#     margin-top: 10px;
# }


# /* Result Card specifics */
# .result-card {
#     min-height: 400px; /* Taller card to match image card */
#     display: flex;
#     flex-direction: column;
#     justify-content: flex-start;
# }

# .result-card h3 {
#     color: #1a73e8 !important; /* Primary blue for result subheaders */
#     border-bottom: none; /* Removed separator line */
#     padding-bottom: 0px;
#     margin-top: 10px;
#     margin-bottom: 10px;
# }

# /* Metric styling */
# .stMetric {
#     background-color: #f8f9fa; /* Lighter background for metrics */
#     border-radius: 8px;
#     padding: 10px 15px;
#     margin-bottom: 10px;
#     border: 1px solid #eee;
# }
# .stMetric label {
#     font-weight: 500;
#     color: #555;
# }
# .stMetric .stMetricValue {
#     font-size: 1.5rem; /* Larger metric values */
#     font-weight: 600;
#     color: #1a73e8; /* Blue for metric values */
# }
# .stMetric .stMetricDelta {
#     color: #333; /* For confidence, keep neutral */
# }


# /* Highlight for the severity result */
# .severity-level {
#     font-size: 1.5rem; /* Slightly larger */
#     font-weight: 700; /* Bolder */
#     margin-top: 20px;
#     padding: 15px;
#     border-radius: 8px;
#     background-color: #e3f2fd; /* Very light blue background for the final result box */
#     text-align: center;
# }
# .severity-level strong {
#     font-size: 1.8rem;
# }

# /* Color coding for severity */
# .NoDR { background-color: #e6ffe6; color: #28a745; border: 2px solid #28a745; } /* Green */
# .MildDR { background-color: #fffbe6; color: #ffc107; border: 2px solid #ffc107; } /* Gold */
# .ModerateDR { background-color: #fff0e6; color: #fd7e14; border: 2px solid #fd7e14; } /* Orange */
# .SevereDR { background-color: #ffe6e6; color: #dc3545; border: 2px solid #dc3545; } /* Red */
# .ProliferativeDR { background-color: #f3e9ff; color: #6f42c1; border: 2px solid #6f42c1; } /* Purple/Dark Red */

# /* --- Streamlit Component Overrides --- */
# /* Buttons */
# .stButton>button {
#     background-color: #1a73e8; /* Blue background */
#     color: white;
#     border-radius: 8px;
#     border: none;
#     padding: 10px 20px;
#     font-size: 1rem;
#     font-weight: 600;
#     margin-top: 10px; /* Adjust button spacing */
#     margin-bottom: 10px;
#     transition: background-color 0.3s ease;
# }
# .stButton>button:hover {
#     background-color: #0d47a1; /* Darker blue on hover */
#     color: white;
# }

# /* File uploader and Camera input styling */
# .stFileUploader, .stCameraInput {
#     border: 2px dashed #a0c4ff; /* Dashed blue border */
#     border-radius: 10px;
#     padding: 20px;
#     background-color: #e9f2ff; /* Light blue background */
#     margin-top: 15px;
#     margin-bottom: 15px;
# }

# /* Info and Warning boxes */
# .stAlert {
#     border-radius: 8px;
# }

# /* Sidebar styling */
# .css-1d391kg { /* Target sidebar main container */
#     background-color: #e9f2ff; /* Light blue for sidebar */
#     border-right: 1px solid #d0e4ff;
#     box-shadow: 2px 0 10px rgba(0,0,0,0.05);
# }
# .css-1d391kg .stSidebar > div:first-child {
#     padding-top: 2rem; /* Add some top padding to sidebar content */
# }
# .sidebar .st-ag { /* Target sidebar text */
#     color: #333;
# }
# .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
#     color: #1a73e8; /* Blue headers in sidebar */
# }
# .stSidebar .stInfo {
#     background-color: #dbeaff;
#     color: #1a73e8;
#     border-color: #a7c9f5;
# }


# /* Progress bar styling (if using) */
# .stProgress > div > div > div > div {
#     background-color: #1a73e8; /* Blue progress bar */
# }

# /* Hide Streamlit footer and any remaining Streamlit UI components */
# #MainMenu { visibility: hidden; }
# footer { visibility: hidden; }
# header { visibility: hidden; }

# /* FIX FOR EXTRA BOXES AND WHITE BARS: 
#    Collapse empty elements that Streamlit might insert. This is often what 
#    causes unexpected white space or boxes (the "white bars"). */
# div[data-testid="stVerticalBlock"] > div > div:empty,
# div[data-testid="stHorizontalBlock"] > div > div:empty {
#     min-height: 0px !important;
#     height: 0px !important;
#     padding: 0px !important;
#     margin: 0px !important;
#     /* Also target the internal container that might hold the empty div */
#     div[data-testid="stVerticalBlock"] > div:empty,
#     div[data-testid="stHorizontalBlock"] > div:empty {
#         min-height: 0px !important;
#         height: 0px !important;
#         padding: 0px !important;
#         margin: 0px !important;
#     }
# }

# /* Target and collapse the specific empty container generated by Streamlit's layout engine */
# [data-testid="stVerticalBlock"] > div:first-child:empty,
# [data-testid="stHorizontalBlock"] > div:first-child:empty,
# [data-testid="stVerticalBlock"] > div:last-child:empty,
# [data-testid="stHorizontalBlock"] > div:last-child:empty {
#     height: 0px !important;
#     min-height: 0px !important;
#     padding: 0px !important;
#     margin: 0px !important;
# }


# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

# # ==========================================================
# # 2. API Interaction Functions
# # ==========================================================

# def call_retinal_classifier(image_file):
#     """Calls the Flask API for Model 1 (Retina vs Non-Retina).
#     Returns a dictionary or a safe default error dictionary."""
#     url = f"{API_BASE_URL}/upload"
#     files = {'image': ('input.png', image_file, 'image/png')}
    
#     try:
#         response = requests.post(url, files=files)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.ConnectionError:
#         st.error(f"Connection Error: Could not connect to the API at {API_BASE_URL}. Ensure your Flask app is running.")
#         return {"message": "API Connection Failed", "confidence": 0.0}
#     except requests.exceptions.HTTPError as e:
#         st.error(f"HTTP Error: The server returned an error: {e}")
#         return {"message": "Server Error", "confidence": 0.0}
#     except Exception as e:
#         st.error(f"An unexpected error occurred during the API call: {e}")
#         return {"message": "Unknown API Error", "confidence": 0.0}

# def call_severity_classifier(image_file):
#     """Calls the Flask API for Model 2 (DR Severity).
#     Returns a dictionary or a safe default error dictionary."""
#     url = f"{API_BASE_URL}/uploadS"
#     files = {'image': ('input.png', image_file, 'image/png')}

#     try:
#         response = requests.post(url, files=files)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.ConnectionError:
#         # Error handled in the first function, keeping simple here
#         return {"message": "API Connection Failed", "confidence": 0.0}
#     except requests.exceptions.HTTPError as e:
#         return {"message": "Server Error", "confidence": 0.0}
#     except Exception as e:
#         st.error(f"An unexpected error occurred during the API call: {e}")
#         return {"message": "Unknown API Error", "confidence": 0.0}

# # ==========================================================
# # 3. Streamlit UI Layout
# # ==========================================================

# # Use a container for the main title to apply card-like styling
# st.markdown("<div class='stCard'>", unsafe_allow_html=True)
# st.title("👁️ Diabetic Retinopathy AI Diagnostic Tool")
# st.markdown("A demonstration interface for deep learning models used in ophthalmology.")
# st.markdown("</div>", unsafe_allow_html=True)

# # Sidebar for information
# with st.sidebar:
#     st.header("About This Demo")
#     st.info(
#         "This application demonstrates a two-stage process for diagnosing Diabetic Retinopathy (DR): "
#         "first, filtering non-retinal images, and second, classifying the severity level."
#     )
#     st.markdown(f"**API Endpoint:** `{API_BASE_URL}`")
#     st.markdown("---")
#     st.subheader("Classification Models")
#     st.markdown(
#         """
#         - **Model 1 (Stage 1):** Retina vs. Non-Retina
#         - **Model 2 (Stage 2):** DR Severity (No DR, Mild, Moderate, Severe, Proliferative)
#         """
#     )
#     st.markdown("---")
#     st.markdown("Developed using Streamlit, Flask, and TensorFlow.")


# # Main content area - Image Input Selection
# st.markdown("<div class='input-card'>", unsafe_allow_html=True)
# st.subheader("Select Image Input Method")

# image_input = None

# if st.session_state.input_mode is None:
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("⬆️ Upload an Image File", use_container_width=True):
#             st.session_state.input_mode = "file"
#             st.rerun()
#     with col2:
#         if st.button("📸 Capture with Camera", use_container_width=True):
#             st.session_state.input_mode = "camera"
#             st.rerun()

# elif st.session_state.input_mode == "file":
#     st.info("Upload Mode Selected. Click 'Change Input' to switch to camera.")
#     col_input, col_reset = st.columns([4, 1])
#     with col_input:
#         image_input = st.file_uploader(
#             "Choose an image file...", 
#             type=["jpg", "jpeg", "png"]
#         )
#     with col_reset:
#         st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
#         if st.button("Change Input", key="reset_file", use_container_width=True):
#             st.session_state.input_mode = None
#             st.rerun()

# elif st.session_state.input_mode == "camera":
#     st.info("Camera Mode Selected. The camera is active below. Click 'Change Input' to switch to file upload.")
#     col_input, col_reset = st.columns([4, 1])
#     with col_input:
#         image_input = st.camera_input("Take a photo of the fundus image")
#     with col_reset:
#         st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
#         if st.button("Change Input", key="reset_camera", use_container_width=True):
#             st.session_state.input_mode = None
#             st.rerun()
# st.markdown("</div>", unsafe_allow_html=True) # Close input-card div


# # Diagnosis Logic starts here, contingent on an image being provided
# if image_input is not None:
#     # 1. Read the image for display using PIL
#     image = Image.open(image_input)
    
#     # --- CRITICAL FIX: Reset the stream pointer ---
#     # After PIL.Image.open reads the stream, the pointer is at the end. 
#     # We must reset it to 0 before reading the bytes for the API calls.
#     image_input.seek(0)
    
#     # 2. Read the complete image bytes for API calls
#     image_bytes = image_input.read()

#     # --- THREE CARD LAYOUT ---
#     col_image, col_retina_result, col_severity_result = st.columns(3)

#     # CARD 1: Image Display
#     with col_image:
#         st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
#         st.subheader("1. Input Image")
#         # Display the actual uploaded image
#         st.image(image, use_container_width=True)
#         st.caption("Image uploaded for analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     # Process the image through the API and display results in cards 2 & 3
    
#     # Placeholder for dynamic content to use during loading
#     retina_card_placeholder = col_retina_result.empty()
#     severity_card_placeholder = col_severity_result.empty()


#     # --- Loading Indicator ---
#     with st.spinner('Analyzing image with AI models...'):
#         # --- Stage 1: Retina Check ---
#         retinal_check_result = call_retinal_classifier(image_bytes)
#         result_label = retinal_check_result.get("message", "N/A")
#         confidence_score = retinal_check_result.get("confidence", 0.0)

#         # CARD 2: Retinal Prediction
#         with retina_card_placeholder.container():
#             st.markdown("<div class='result-card'>", unsafe_allow_html=True)
#             st.subheader("2. Retina Check (Stage 1)")

#             # Check for error message from the API functions
#             if result_label in ["API Connection Failed", "Server Error", "Unknown API Error"]:
#                 st.error(f"❌ Stage 1 Failed: {result_label}. Check server.")
#             else:
#                 st.markdown(f"**Classification:** <span style='font-size: 1.5rem; font-weight: 600; color: #1a73e8;'>{result_label.upper()}</span>", unsafe_allow_html=True)
#                 st.metric(
#                     label="Confidence", 
#                     value=f"{confidence_score*100:.2f}%" 
#                 )
#                 if result_label.lower() == 'retinal':
#                     st.success("✅ Image is valid for DR screening.")
#                 else:
#                     st.error("❌ Image is non-retinal. DR screening aborted.")

#             st.markdown("</div>", unsafe_allow_html=True)

#         # CARD 3: DR Severity Check (Only if retinal)
#         with severity_card_placeholder.container():
#             st.markdown("<div class='result-card'>", unsafe_allow_html=True)
#             st.subheader("3. DR Severity (Stage 2)")

#             if result_label.lower() == 'retinal':
                
#                 # --- Stage 2 Execution ---
#                 severity_result = call_severity_classifier(image_bytes)
#                 severity_label = severity_result.get("message", "N/A")
#                 severity_confidence = severity_result.get("confidence", 0.0)

#                 # Check for error message from the API functions
#                 if severity_label in ["API Connection Failed", "Server Error", "Unknown API Error"]:
#                     st.error(f"❌ Stage 2 Failed: {severity_label}")
#                 else:
#                     # Clean up class name for CSS color coding (e.g., 'No DR' -> 'NoDR')
#                     css_class = severity_label.replace(" ", "").replace("-", "")

#                     st.markdown(f"<p class='severity-level {css_class}'>**Diabetic Retinopathy**<br/>Severity: <strong>{severity_label.upper()}</strong></p>", unsafe_allow_html=True)
#                     st.metric(
#                         label="Prediction Confidence", 
#                         value=f"{severity_confidence*100:.2f}%"
#                     )
#                     st.success("✅ Full diagnosis complete.")

#             else:
#                 st.info("Awaiting valid retinal image from Stage 1.")
        
#             st.markdown("</div>", unsafe_allow_html=True)


# else:
#     # --- Initial State Layout (No image uploaded yet) ---
#     # Display placeholder or initial message when no input mode is selected or no file is uploaded/captured
#     if st.session_state.input_mode is None:
#         st.info("⬆️ Select an input method above to begin the diagnostic process.")
    
#     # Placeholder layout (One image card, two empty result cards)
#     col_image, col_retina_placeholder, col_severity_placeholder = st.columns(3)
    
#     with col_image:
#         st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
#         st.subheader("1. Input Image")
#         # Placeholder image, centered inside the card
#         st.image("https://placehold.co/800x400/1a73e8/ffffff?text=Upload+Fundus+Image+To+Start", use_container_width=True) 
#         st.caption("Upload or capture an image above to begin analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     # Empty result card placeholders - NO EXTRA CONTENT/WAVY LINES HERE
#     with col_retina_placeholder:
#         st.markdown("<div class='result-card'>", unsafe_allow_html=True)
#         st.subheader("2. Retina Check (Stage 1)")
#         st.info("Awaiting image upload to start analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     with col_severity_placeholder:
#         st.markdown("<div class='result-card'>", unsafe_allow_html=True)
#         st.subheader("3. DR Severity (Stage 2)")
#         st.info("Awaiting valid result from Stage 1.")
#         st.markdown("</div>", unsafe_allow_html=True)


# st.markdown("---")
# st.caption("Disclaimer: This tool is for demonstration purposes only and is not a substitute for professional medical advice.")

import streamlit as st
import requests
from io import BytesIO
import json
from PIL import Image
import time # For simulating loading

# ==========================================================
# 1. Configuration and CSS Fix & Enhanced Styling
# ==========================================================

# Use the environment variable or default to a local Flask server URL.
# IMPORTANT: Adjust this URL if your Flask backend is hosted elsewhere!
API_BASE_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Diabetic Retinopathy AI Diagnostic",
    layout="wide", # Use wide layout for more horizontal space
    initial_sidebar_state="expanded"
)

# Initialize session state for input mode control
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None

# Enhanced Custom CSS for a professional, card-based look
custom_css = """
<style>
/* --- Core Page & Layout Adjustments --- */
/* FIX: Aggressively remove top padding for the entire app and main sections */
#stApp {
    padding-top: 0px !important; 
    margin-top: 0px !important;
}
section.main {
    padding-top: 0px !important;
    margin-top: 0px !important;
}

/* Ensure the Streamlit header is also compact if it exists */
.stApp > header {
    height: 0px !important; /* Hide Streamlit's default header */
    visibility: hidden;
}

/* General body styling */
body {
    font-family: 'Inter', sans-serif;
    color: #333; /* Darker text for readability */
    background-color: #f0f2f6; /* Light grey background */
}

/* TIGHTEN: Reduce default margins on all headers to minimize vertical space */
h1, h2, h3, h4, h5, h6 {
    margin-top: 0.5em; 
    margin-bottom: 0.5em; 
    color: #212121; /* Darker header color, will be overridden below */
}

/* Streamlit Title specific styling */
.stTitle {
    color: #1a73e8; /* Google Blue equivalent */
    font-weight: 700;
    margin-bottom: 0.2em; /* Reduce space below title */
}

/* UNIFIED HEADING COLOR: Make all primary headings blue */
.stCard h2, .result-card h2, .input-card h2, 
.result-card h3, .stSidebar h3 {
    color: #1a73e8 !important; /* Force all relevant headings to blue */
}


/* Markdown styling (e.g., for subheader descriptions) */
.stMarkdown {
    color: #555;
}

/* --- Card Styling --- */
.stCard, .result-card, .input-card {
    background-color: #ffffff; /* White background for cards */
    border-radius: 12px; /* More rounded corners */
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); /* Stronger, softer shadow */
    padding: 25px; /* More internal padding */
    margin-bottom: 25px; /* Space between cards */
    border: 1px solid #e0e0e0; /* Subtle border */
    min-height: 250px; /* Ensure cards have some minimum height */
}

/* Specific styling for the input method card */
.input-card {
    padding-top: 15px;
    padding-bottom: 15px;
}

/* Styling for the image display card */
.image-display-card {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    padding: 15px; /* Slightly less padding to contain image */
    margin-bottom: 25px;
    border: 1px solid #e0e0e0;
    text-align: center; /* Center image caption */
    min-height: 400px; /* Taller card for the image */
    display: flex;
    flex-direction: column;
    justify-content: flex-start; /* Align content to the top */
    align-items: center;
}
.image-display-card img {
    border-radius: 8px; /* Rounded corners for the image itself */
    max-width: 100%;
    height: auto;
    margin-top: 10px; /* Space below the header */
}
.image-display-card .stCaption {
    color: #777;
    margin-top: 10px;
}


/* Result Card specifics */
.result-card {
    min-height: 400px; /* Taller card to match image card */
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

.result-card h3 {
    color: #1a73e8 !important; /* Primary blue for result subheaders */
    border-bottom: none; /* Removed separator line */
    padding-bottom: 0px;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Metric styling */
.stMetric {
    background-color: #f8f9fa; /* Lighter background for metrics */
    border-radius: 8px;
    padding: 10px 15px;
    margin-bottom: 10px;
    border: 1px solid #eee;
}
.stMetric label {
    font-weight: 500;
    color: #555;
}
.stMetric .stMetricValue {
    font-size: 1.5rem; /* Larger metric values */
    font-weight: 600;
    color: #1a73e8; /* Blue for metric values */
}
.stMetric .stMetricDelta {
    color: #333; /* For confidence, keep neutral */
}


/* Highlight for the severity result */
.severity-level {
    font-size: 1.5rem; /* Slightly larger */
    font-weight: 700; /* Bolder */
    margin-top: 20px;
    padding: 15px;
    border-radius: 8px;
    background-color: #e3f2fd; /* Very light blue background for the final result box */
    text-align: center;
}
.severity-level strong {
    font-size: 1.8rem;
}

/* Color coding for severity */
.NoDR { background-color: #e6ffe6; color: #28a745; border: 2px solid #28a745; } /* Green */
.MildDR { background-color: #fffbe6; color: #ffc107; border: 2px solid #ffc107; } /* Gold */
.ModerateDR { background-color: #fff0e6; color: #fd7e14; border: 2px solid #fd7e14; } /* Orange */
.SevereDR { background-color: #ffe6e6; color: #dc3545; border: 2px solid #dc3545; } /* Red */
.ProliferativeDR { background-color: #f3e9ff; color: #6f42c1; border: 2px solid #6f42c1; } /* Purple/Dark Red */

/* --- Streamlit Component Overrides --- */
/* Buttons */
.stButton>button {
    background-color: #1a73e8; /* Blue background */
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 10px; /* Adjust button spacing */
    margin-bottom: 10px;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #0d47a1; /* Darker blue on hover */
    color: white;
}

/* File uploader and Camera input styling */
.stFileUploader, .stCameraInput {
    border: 2px dashed #a0c4ff; /* Dashed blue border */
    border-radius: 10px;
    padding: 20px;
    background-color: #e9f2ff; /* Light blue background */
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Info and Warning boxes */
.stAlert {
    border-radius: 8px;
}

/* Sidebar styling */
.css-1d391kg { /* Target sidebar main container */
    background-color: #e9f2ff; /* Light blue for sidebar */
    border-right: 1px solid #d0e4ff;
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
}
.css-1d391kg .stSidebar > div:first-child {
    padding-top: 2rem; /* Add some top padding to sidebar content */
}
.sidebar .st-ag { /* Target sidebar text */
    color: #333;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
    color: #1a73e8; /* Blue headers in sidebar */
}
.stSidebar .stInfo {
    background-color: #dbeaff;
    color: #1a73e8;
    border-color: #a7c9f5;
}


/* Progress bar styling (if using) */
.stProgress > div > div > div > div {
    background-color: #1a73e8; /* Blue progress bar */
}

/* Hide Streamlit footer and any remaining Streamlit UI components */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* FIX FOR EXTRA BOXES AND WHITE BARS: 
   Collapse empty Streamlit container wrappers */
div[data-testid^="stHorizontalBlock"] > div:empty,
div[data-testid^="stVerticalBlock"] > div:empty,
/* Target the internal element that holds the content padding within a column */
div[data-testid="column"] > div {
    padding: 0px !important; /* Force padding to zero within the column */
    margin: 0px !important;  /* Force margin to zero within the column */
}
/* Ensure elements that are explicitly empty truly collapse */
div:empty {
    min-height: 0px !important;
    height: 0px !important;
    padding: 0px !important;
    margin: 0px !important;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================================
# 2. API Interaction Functions
# ==========================================================

def call_retinal_classifier(image_file):
    """Calls the Flask API for Model 1 (Retina vs Non-Retina).
    Returns a dictionary or a safe default error dictionary."""
    url = f"{API_BASE_URL}/upload"
    files = {'image': ('input.png', image_file, 'image/png')}
    
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the API at {API_BASE_URL}. Ensure your Flask app is running.")
        return {"message": "API Connection Failed", "confidence": 0.0}
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: The server returned an error: {e}")
        return {"message": "Server Error", "confidence": 0.0}
    except Exception as e:
        st.error(f"An unexpected error occurred during the API call: {e}")
        return {"message": "Unknown API Error", "confidence": 0.0}

def call_severity_classifier(image_file):
    """Calls the Flask API for Model 2 (DR Severity).
    Returns a dictionary or a safe default error dictionary."""
    url = f"{API_BASE_URL}/uploadS"
    files = {'image': ('input.png', image_file, 'image/png')}

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        # Error handled in the first function, keeping simple here
        return {"message": "API Connection Failed", "confidence": 0.0}
    except requests.exceptions.HTTPError as e:
        return {"message": "Server Error", "confidence": 0.0}
    except Exception as e:
        st.error(f"An unexpected error occurred during the API call: {e}")
        return {"message": "Unknown API Error", "confidence": 0.0}

# ==========================================================
# 3. Streamlit UI Layout
# ==========================================================

# Use a container for the main title to apply card-like styling
st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.title("👁️ Diabetic Retinopathy AI Diagnostic Tool")
st.markdown("A demonstration interface for deep learning models used in ophthalmology.")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for information
with st.sidebar:
    st.header("About This Demo")
    st.info(
        "This application demonstrates a two-stage process for diagnosing Diabetic Retinopathy (DR): "
        "first, filtering non-retinal images, and second, classifying the severity level."
    )
    st.markdown(f"**API Endpoint:** `{API_BASE_URL}`")
    st.markdown("---")
    st.subheader("Classification Models")
    st.markdown(
        """
        - **Model 1 (Stage 1):** Retina vs. Non-Retina
        - **Model 2 (Stage 2):** DR Severity (No DR, Mild, Moderate, Severe, Proliferative)
        """
    )
    st.markdown("---")
    st.markdown("Developed using Streamlit, Flask, and TensorFlow.")


# Main content area - Image Input Selection
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.subheader("Select Image Input Method")

image_input = None

if st.session_state.input_mode is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬆️ Upload an Image File", use_container_width=True):
            st.session_state.input_mode = "file"
            st.rerun()
    with col2:
        if st.button("📸 Capture with Camera", use_container_width=True):
            st.session_state.input_mode = "camera"
            st.rerun()

elif st.session_state.input_mode == "file":
    st.info("Upload Mode Selected. Click 'Change Input' to switch to camera.")
    col_input, col_reset = st.columns([4, 1])
    with col_input:
        image_input = st.file_uploader(
            "Choose an image file...", 
            type=["jpg", "jpeg", "png"]
        )
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
        if st.button("Change Input", key="reset_file", use_container_width=True):
            st.session_state.input_mode = None
            st.rerun()

elif st.session_state.input_mode == "camera":
    st.info("Camera Mode Selected. The camera is active below. Click 'Change Input' to switch to file upload.")
    col_input, col_reset = st.columns([4, 1])
    with col_input:
        image_input = st.camera_input("Take a photo of the fundus image")
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer for alignment
        if st.button("Change Input", key="reset_camera", use_container_width=True):
            st.session_state.input_mode = None
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True) # Close input-card div


# Diagnosis Logic starts here, contingent on an image being provided
if image_input is not None:
    # 1. Read the image for display using PIL
    image = Image.open(image_input)
    
    # --- CRITICAL FIX: Reset the stream pointer ---
    # After PIL.Image.open reads the stream, the pointer is at the end. 
    # We must reset it to 0 before reading the bytes for the API calls.
    image_input.seek(0)
    
    # 2. Read the complete image bytes for API calls
    image_bytes = image_input.read()

    # --- THREE CARD LAYOUT ---
    col_image, col_retina_result, col_severity_result = st.columns(3)

    # CARD 1: Image Display
    with col_image:
        st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
        st.subheader("1. Input Image")
        # Display the actual uploaded image
        st.image(image, use_container_width=True)
        st.caption("Image uploaded for analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Process the image through the API and display results in cards 2 & 3
    
    # Placeholder for dynamic content to use during loading
    retina_card_placeholder = col_retina_result.empty()
    severity_card_placeholder = col_severity_result.empty()


    # --- Loading Indicator ---
    with st.spinner('Analyzing image with AI models...'):
        # --- Stage 1: Retina Check ---
        retinal_check_result = call_retinal_classifier(image_bytes)
        result_label = retinal_check_result.get("message", "N/A")
        confidence_score = retinal_check_result.get("confidence", 0.0)

        # CARD 2: Retinal Prediction
        with retina_card_placeholder.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("2. Retina Check (Stage 1)")

            # Check for error message from the API functions
            if result_label in ["API Connection Failed", "Server Error", "Unknown API Error"]:
                st.error(f"❌ Stage 1 Failed: {result_label}. Check server.")
            else:
                st.markdown(f"**Classification:** <span style='font-size: 1.5rem; font-weight: 600; color: #1a73e8;'>{result_label.upper()}</span>", unsafe_allow_html=True)
                st.metric(
                    label="Confidence", 
                    value=f"{confidence_score*100:.2f}%" 
                )
                if result_label.lower() == 'retinal':
                    st.success("✅ Image is valid for DR screening.")
                else:
                    st.error("❌ Image is non-retinal. DR screening aborted.")

            st.markdown("</div>", unsafe_allow_html=True)

        # CARD 3: DR Severity Check (Only if retinal)
        with severity_card_placeholder.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("3. DR Severity (Stage 2)")

            if result_label.lower() == 'retinal':
                
                # --- Stage 2 Execution ---
                severity_result = call_severity_classifier(image_bytes)
                severity_label = severity_result.get("message", "N/A")
                severity_confidence = severity_result.get("confidence", 0.0)

                # Check for error message from the API functions
                if severity_label in ["API Connection Failed", "Server Error", "Unknown API Error"]:
                    st.error(f"❌ Stage 2 Failed: {severity_label}")
                else:
                    # Clean up class name for CSS color coding (e.g., 'No DR' -> 'NoDR')
                    css_class = severity_label.replace(" ", "").replace("-", "")

                    st.markdown(f"<p class='severity-level {css_class}'>**Diabetic Retinopathy**<br/>Severity: <strong>{severity_label.upper()}</strong></p>", unsafe_allow_html=True)
                    st.metric(
                        label="Prediction Confidence", 
                        value=f"{severity_confidence*100:.2f}%"
                    )
                    st.success("✅ Full diagnosis complete.")

            else:
                st.info("Awaiting valid retinal image from Stage 1.")
        
            st.markdown("</div>", unsafe_allow_html=True)


else:
    # --- Initial State Layout (No image uploaded yet) ---
    # Display placeholder or initial message when no input mode is selected or no file is uploaded/captured
    if st.session_state.input_mode is None:
        st.info("⬆️ Select an input method above to begin the diagnostic process.")
    
    # Placeholder layout (One image card, two empty result cards)
    col_image, col_retina_placeholder, col_severity_placeholder = st.columns(3)
    
    with col_image:
        st.markdown("<div class='image-display-card'>", unsafe_allow_html=True)
        st.subheader("1. Input Image")
        # Placeholder image, centered inside the card
        st.image("https://placehold.co/800x400/1a73e8/ffffff?text=Upload+Fundus+Image+To+Start", use_container_width=True) 
        st.caption("Upload or capture an image above to begin analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Empty result card placeholders - NO EXTRA CONTENT/WAVY LINES HERE
    with col_retina_placeholder:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("2. Retina Check (Stage 1)")
        st.info("Awaiting image upload to start analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_severity_placeholder:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("3. DR Severity (Stage 2)")
        st.info("Awaiting valid result from Stage 1.")
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.caption("Disclaimer: This tool is for demonstration purposes only and is not a substitute for professional medical advice.")