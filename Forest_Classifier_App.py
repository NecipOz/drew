import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Forest Classification Predictor",
    page_icon="🌲",
    layout="wide"
)

# Title and header
st.title("🌲 Forest Cover Type Classification")
html_temp = """
<div style="background-color:#2E8B57;padding:10px;border-radius:10px">
<h2 style="color:white;text-align:center;">Single Forest Classification Predictor</h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

# Define the exact features used in your model
SELECTED_FEATURES = [
    'elevation',
    'horizontal_distance_to_roadways',
    'horizontal_distance_to_fire_points',
    'horizontal_distance_to_hydrology',
    'wilderness_area4',
    'hillshade_9am',
    'vertical_distance_to_hydrology',
    'wilderness_area1',
    'hillshade_noon',
    'wilderness_area3',
    'aspect'
]

# REAL feature ranges (what users understand) and their scaled equivalents
REAL_FEATURE_RANGES = {
    'elevation': (1863, 3849, 2752),
    'horizontal_distance_to_roadways': (0, 6890, 1316),
    'horizontal_distance_to_fire_points': (0, 6993, 1256),
    'horizontal_distance_to_hydrology': (0, 1343, 180),
    'wilderness_area4': (0, 1, 0),
    'hillshade_9am': (58, 254, 220),
    'vertical_distance_to_hydrology': (-146, 554, 32),
    'wilderness_area1': (0, 1, 0),
    'hillshade_noon': (99, 254, 223),
    'wilderness_area3': (0, 1, 0),
    'aspect': (0, 360, 126),
}

# Mapping from real values to scaled values (approximate conversion factors)
# These are rough estimates - you should get the exact scaler parameters from your notebook
SCALING_PARAMS = {
    'elevation': {'mean': 2752, 'std': 417},
    'horizontal_distance_to_roadways': {'mean': 1316, 'std': 1500},
    'horizontal_distance_to_fire_points': {'mean': 1256, 'std': 1400},
    'horizontal_distance_to_hydrology': {'mean': 180, 'std': 200},
    'hillshade_9am': {'mean': 220, 'std': 30},
    'vertical_distance_to_hydrology': {'mean': 32, 'std': 100},
    'hillshade_noon': {'mean': 223, 'std': 25},
    'aspect': {'mean': 126, 'std': 100},
}

def real_to_scaled(feature_name, real_value):
    """Convert real value to scaled value"""
    if feature_name in SCALING_PARAMS:
        params = SCALING_PARAMS[feature_name]
        return (real_value - params['mean']) / params['std']
    else:
        return real_value  # For binary features

# Forest type mapping
FOREST_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine", 
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# Load models
@st.cache_resource
def load_models():
    """Load the saved model"""
    try:
        with open("forest_cover_catboost_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully")
            
    except FileNotFoundError:
        st.error("❌ forest_cover_catboost_classifier.pkl not found")
        model = None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        model = None
    
    return model

# Create sidebar for inputs
st.sidebar.title('🌲 Enter Forest Characteristics')
st.sidebar.info("Enter real-world values - the app will automatically scale them for the model")

# Create input fields for all features
input_values = {}
scaled_values = {}

# Continuous features
st.sidebar.subheader("📏 Terrain Measurements")
for feature in ['elevation', 'horizontal_distance_to_roadways', 
                'horizontal_distance_to_fire_points', 'horizontal_distance_to_hydrology',
                'vertical_distance_to_hydrology']:
    if feature in REAL_FEATURE_RANGES:
        min_val, max_val, default = REAL_FEATURE_RANGES[feature]
        
        # Add appropriate units to labels
        if 'distance' in feature or 'elevation' in feature:
            unit = " (meters)"
        else:
            unit = ""
        
        # Create readable label
        label = feature.replace('_', ' ').title() + unit
        
        input_values[feature] = st.sidebar.slider(
            label,
            min_val, max_val, default
        )
        
        # Convert to scaled value
        scaled_values[feature] = real_to_scaled(feature, input_values[feature])

# Aspect (special case - degrees)
feature = 'aspect'
if feature in REAL_FEATURE_RANGES:
    min_val, max_val, default = REAL_FEATURE_RANGES[feature]
    label = "Aspect (degrees)"
    
    input_values[feature] = st.sidebar.slider(
        label,
        min_val, max_val, default
    )
    scaled_values[feature] = real_to_scaled(feature, input_values[feature])

# Hillshade features
st.sidebar.subheader("☀️ Hillshade Values")
for feature in ['hillshade_9am', 'hillshade_noon']:
    if feature in REAL_FEATURE_RANGES:
        min_val, max_val, default = REAL_FEATURE_RANGES[feature]
        label = feature.replace('_', ' ').title()
        input_values[feature] = st.sidebar.slider(
            label,
            min_val, max_val, default
        )
        scaled_values[feature] = real_to_scaled(feature, input_values[feature])

# Wilderness areas (binary features)
st.sidebar.subheader("🏞️ Wilderness Areas")
st.sidebar.info("Select only one wilderness area")

# Use radio buttons to ensure only one is selected
wilderness_selection = st.sidebar.radio(
    "Which wilderness area?",
    ["None", "Wilderness Area 1", "Wilderness Area 3", "Wilderness Area 4"]
)

# Set binary values based on selection
input_values['wilderness_area1'] = 1 if wilderness_selection == "Wilderness Area 1" else 0
input_values['wilderness_area3'] = 1 if wilderness_selection == "Wilderness Area 3" else 0
input_values['wilderness_area4'] = 1 if wilderness_selection == "Wilderness Area 4" else 0

# Binary features don't need scaling
scaled_values['wilderness_area1'] = input_values['wilderness_area1']
scaled_values['wilderness_area3'] = input_values['wilderness_area3']
scaled_values['wilderness_area4'] = input_values['wilderness_area4']

# Create feature vector in the exact order specified (using SCALED values)
feature_vector = [scaled_values[feature] for feature in SELECTED_FEATURES]

# Display current configuration
st.header("Current Forest Configuration:")

# Create a more readable display
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏔️ Terrain Features")
    terrain_data = {
        "Elevation": f"{input_values['elevation']} m",
        "Aspect": f"{input_values['aspect']}°",
        "Distance to Roadways": f"{input_values['horizontal_distance_to_roadways']} m",
        "Distance to Fire Points": f"{input_values['horizontal_distance_to_fire_points']} m",
        "Distance to Water (Horizontal)": f"{input_values['horizontal_distance_to_hydrology']} m",
        "Distance to Water (Vertical)": f"{input_values['vertical_distance_to_hydrology']} m"
    }
    for key, value in terrain_data.items():
        st.write(f"**{key}:** {value}")

with col2:
    st.subheader("☀️ Light & Area Features")
    st.write(f"**Hillshade at 9am:** {input_values['hillshade_9am']}")
    st.write(f"**Hillshade at Noon:** {input_values['hillshade_noon']}")
    st.write(f"**Wilderness Area:** {wilderness_selection}")

# Show the exact feature vector that will be used
with st.expander("🔍 View Feature Values (Technical Details)"):
    feature_df = pd.DataFrame({
        'Feature': SELECTED_FEATURES,
        'Real Value': [input_values[feature] for feature in SELECTED_FEATURES],
        'Scaled Value': feature_vector
    })
    st.dataframe(feature_df, hide_index=True)
    
    # Add copy-paste friendly format for testing in notebook
    st.subheader("📋 Copy for Notebook Testing (Scaled Values):")
    feature_values_str = str(feature_vector)
    st.code(f"test_features = np.array([{feature_values_str}])")
    st.code(f"prediction = cat_grid_3.predict(test_features) + 1")
    st.code(f"print('Prediction:', prediction[0])")

# Prediction section
st.subheader("🔮 Press Predict to classify the forest type")

if st.button("Predict Forest Type", type="primary", use_container_width=True):
    model = load_models()
    
    if model is not None:
        try:
            # Create DataFrame with the exact feature names
            input_df = pd.DataFrame([feature_vector], columns=SELECTED_FEATURES)
            
            # Features are already scaled from sliders, so use directly
            input_scaled = input_df.values
            st.success("✅ Using scaled features directly from sliders")
            
            # Make prediction
            prediction_raw = model.predict(input_scaled)
            
            # Extract the actual prediction value
            if isinstance(prediction_raw, np.ndarray):
                if prediction_raw.ndim > 1:
                    prediction = prediction_raw[0, 0]  # For 2D arrays
                else:
                    prediction = prediction_raw[0]  # For 1D arrays
            else:
                prediction = prediction_raw
                
            # Convert to Python int and add 1 to convert from 0-6 to 1-7
            prediction = int(prediction) + 1
            
            # Debug: Show the raw prediction
            st.info(f"🔍 Model predicted class: {prediction}")
            
            # Create columns for centered display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Display the main prediction
                st.success(f"### 🌲 Predicted Forest Type:")
                st.success(f"# **{FOREST_TYPES.get(prediction, f'Type {prediction}')}**")
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
                
                # Find confidence
                max_prob = max(probabilities)
                confidence = max_prob * 100
                
                # Display confidence
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.info(f"**Confidence Level:** {confidence:.1f}%")
                
                # Create probability chart
                st.subheader("📊 Prediction Probabilities:")
                
                # For CatBoost with classes 1-7, the probability array indices 0-6 
                # correspond to forest types 1-7
                prob_data = []
                
                for i, prob in enumerate(probabilities):
                    # Index 0 = Forest Type 1, Index 1 = Forest Type 2, etc.
                    forest_type_id = i + 1
                    prob_data.append({
                        'Forest Type': FOREST_TYPES[forest_type_id],
                        'Probability (%)': float(prob) * 100
                    })
                
                prob_df = pd.DataFrame(prob_data)
                prob_df = prob_df.sort_values('Probability (%)', ascending=True)
                
                # Create horizontal bar chart
                st.bar_chart(prob_df.set_index('Forest Type')['Probability (%)'])
                
                # Show top 3 predictions
                st.subheader("🏆 Top 3 Most Likely Forest Types:")
                top_3 = prob_df.nlargest(3, 'Probability (%)')
                for idx, row in top_3.iterrows():
                    st.write(f"{row['Forest Type']}: {row['Probability (%)']:.1f}%")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            
            # Show debugging information
            with st.expander("🐛 Debugging Information"):
                st.write("**Error details:**")
                st.code(str(e))
                st.write(f"**Feature vector length:** {len(feature_vector)}")
                st.write(f"**Expected features:** {len(SELECTED_FEATURES)}")
                st.write(f"**Input shape:** {input_df.shape if 'input_df' in locals() else 'N/A'}")
                if model and hasattr(model, 'n_features_in_'):
                    st.write(f"**Model expects:** {model.n_features_in_} features")
    else:
        st.error("❌ Model not loaded. Please ensure 'forest_model.pkl' is in the same directory as this app.")

# Information sections
with st.expander("ℹ️ About Forest Types"):
    forest_descriptions = {
        1: "**Spruce/Fir** - Typically found at higher elevations in cool, moist environments. Common in subalpine regions.",
        2: "**Lodgepole Pine** - A pioneer species that often colonizes areas after forest fires. Grows in dense stands.",
        3: "**Ponderosa Pine** - Prefers warmer, drier sites at lower elevations. Has thick, fire-resistant bark.",
        4: "**Cottonwood/Willow** - Found near water sources like streams and rivers. Requires moist soil conditions.",
        5: "**Aspen** - Deciduous trees that often grow in large clonal colonies. Prefer moist soils and cooler temperatures.",
        6: "**Douglas-fir** - Versatile species that can grow in various conditions. Important timber tree.",
        7: "**Krummholz** - Stunted, wind-blown trees found at tree line. Adapted to harsh alpine conditions."
    }
    
    for type_id, description in forest_descriptions.items():
        st.write(f"{type_id}. {description}")

with st.expander("📊 About the Features"):
    st.write("""
    **Your model uses these 11 carefully selected features:**
    
    **Real-world values are converted to scaled values for the model:**
    
    1. **Elevation** - Height above sea level (1863-3849 meters)
    2. **Horizontal Distance to Roadways** - Distance to nearest road (0-6890 meters)
    3. **Horizontal Distance to Fire Points** - Distance to fire history locations (0-6993 meters)
    4. **Horizontal Distance to Hydrology** - Distance to water sources (0-1343 meters)
    5. **Vertical Distance to Hydrology** - Elevation difference to water (-146 to 554 meters)
    6. **Hillshade at 9am** - Morning sun exposure (58-254)
    7. **Hillshade at Noon** - Midday sun exposure (99-254)
    8. **Aspect** - Compass direction of slope face (0-360 degrees)
    9. **Wilderness Areas 1, 3, and 4** - Different wilderness designations (0 or 1)
    
    The app automatically converts these real values to the scaled format your model expects.
    """)

with st.expander("🎛️ How to Use the App"):
    st.write("""
    **Using Real-World Values:**
    
    1. **Elevation**: Enter the actual elevation in meters (e.g., 2500m for mid-elevation)
    2. **Distances**: Enter actual distances in meters to roads, fire points, and water
    3. **Hillshade**: Values from 0-255 representing sun exposure intensity
    4. **Aspect**: Compass direction in degrees (0=North, 90=East, 180=South, 270=West)
    5. **Wilderness Area**: Choose the appropriate wilderness designation
    
    **The app will:**
    - Show you the real values you entered
    - Automatically convert them to scaled values for the model
    - Make predictions based on the scaled values
    - Display both real and scaled values in the technical details
    
    **Tips for realistic combinations:**
    - Higher elevations (2800m+) often have Spruce/Fir or Krummholz
    - Lower elevations (2000m-) may have Ponderosa Pine
    - Areas close to water often have Cottonwood/Willow
    - Different aspects affect sun exposure and moisture
    """)

# Footer
st.markdown("---")
st.markdown("🌲 Forest Cover Type Prediction App | Real Values Automatically Scaled")