import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Fleet Maintenance Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .prediction-yes {
        background-color: #ffcdd2;
        border: 1px solid #e57373;
    }
    .prediction-no {
        background-color: #c8e6c9;
        border: 1px solid #81c784;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 2px solid #1E88E5;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    #MainMenu {
        visibility: hidden;
        height: 0%;
    }
    footer {
        visibility: hidden;
        height: 0%;
    }
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 300px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model and columns
@st.cache_resource
def load_model():
    try:
        with open('models/fleet_maintenance_prediction_model_cpu', 'rb') as f:
            model = pickle.load(f)
        
        with open('columns.json', 'r') as f:
            columns = json.load(f)['data_columns']
        
        return model, columns
    except Exception as e:
        st.error(f"Error loading model or columns: {e}")
        return None, None

# Function to load sample data (if available)
@st.cache_data
def load_sample_data():
    try:
        data = pd.read_csv("data/balanced_fleet_maintenance_schedule_2.csv")
        return data
    except:
        return None

# Function to create a gauge chart for prediction probability
def create_gauge_chart(probability, threshold=0.5):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Maintenance Probability (%)", 'font': {'size': 24}},
        delta = {'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': 'lightgreen'},
                {'range': [threshold * 100, 100], 'color': 'lightcoral'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100}}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial", size=12)
    )
    return fig

# Function to create a radar chart for input features
def create_radar_chart(input_data, feature_importance):
    # Get top features by importance
    top_features_idx = np.argsort(feature_importance)[-8:]  # Top 8 features
    top_features = [list(input_data.columns)[i] for i in top_features_idx]
    top_importance = [feature_importance[i] for i in top_features_idx]
    
    # Normalize the input values for these features
    values = []
    for feature in top_features:
        value = input_data[feature].values[0]
        # Simple min-max scaling to 0-1 range for visualization
        # In a real app, you'd use proper scaling based on your training data
        if feature == "mileage_km":
            scaled = min(value / 100000, 1)  # Assuming 100,000 km is high
        elif feature == "engine_hours":
            scaled = min(value / 5000, 1)    # Assuming 5,000 hours is high
        elif feature == "number_of_trips":
            scaled = min(value / 1000, 1)    # Assuming 1,000 trips is high
        elif feature == "breakdown_incidents":
            scaled = min(value / 5, 1)       # Assuming 5 incidents is high
        elif feature == "part_replacements":
            scaled = min(value / 10, 1)      # Assuming 10 replacements is high
        elif feature == "service_history":
            scaled = min(value / 10, 1)      # Assuming 10 services is high
        else:
            scaled = value / 100             # Generic scaling
        values.append(scaled)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=top_features,
        fill='toself',
        name='Vehicle Metrics',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add importance as another trace
    fig.add_trace(go.Scatterpolar(
        r=[imp for imp in top_importance],
        theta=top_features,
        fill='toself',
        name='Feature Importance',
        line_color='rgb(255, 127, 14)',
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Vehicle Metrics vs. Feature Importance",
        height=400
    )
    
    return fig

# Main function
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/truck--v1.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction Dashboard", "Data Explorer", "About"])
    
    # Load model and columns
    model, columns = load_model()
    
    if model is None or columns is None:
        st.error("Please make sure the model file and columns.json are in the same directory as this app.")
        st.info("You can generate these files by running your training script.")
        return
    
    # Load sample data if available
    sample_data = load_sample_data()
    
    if page == "Prediction Dashboard":
        # Header
        st.markdown('<p class="main-header">Fleet Maintenance Prediction System</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Predict maintenance requirements based on vehicle data</p>', unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Batch Prediction", "🔍 What-If Analysis"])
        
        with tab1:
            st.markdown("### Enter Vehicle Details")
            
            # Create two columns for input fields
            col1, col2 = st.columns(2)
            
            with col1:
                mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
                fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.5, step=0.1)
                engine_hours = st.number_input("Engine Hours", min_value=0, value=1200)
                trips = st.number_input("Number of Trips", min_value=0, value=500)
                engine_temp = st.number_input("Engine Temperature (°C)", min_value=0, value=90)
                tire_pressure = st.number_input("Tire Pressure (PSI)", min_value=0.0, value=32.5, step=0.1)
                battery_voltage = st.number_input("Battery Voltage (V)", min_value=0.0, value=12.5, step=0.1)
            
            with col2:
                oil_level = st.number_input("Oil Level (L)", min_value=0.0, value=4.5, step=0.1)
                vibration = st.number_input("Vibration Units", min_value=0.0, value=2.5, step=0.1)
                noise_level = st.number_input("Noise Level (dB)", min_value=0, value=75)
                breakdown_incidents = st.number_input("Breakdown Incidents", min_value=0, value=1)
                part_replacements = st.number_input("Part Replacements", min_value=0, value=2)
                service_history = st.number_input("Service History", min_value=0, value=3)
            
            # Create a dictionary with user inputs
            input_data = {
                "mileage_km": mileage,
                "fuel_consumption_l_per_100km": fuel_consumption,
                "engine_hours": engine_hours,
                "number_of_trips": trips,
                "engine_temperature_c": engine_temp,
                "tire_pressure_psi": tire_pressure,
                "battery_voltage_v": battery_voltage,
                "oil_level_l": oil_level,
                "vibration_units": vibration,
                "noise_level_db": noise_level,
                "breakdown_incidents": breakdown_incidents,
                "part_replacements": part_replacements,
                "service_history": service_history
            }
            
            # Create a DataFrame with user inputs
            input_df = pd.DataFrame([input_data])
            
            # Ensure all expected columns are present
            for col in columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[columns]
            
            # Predict button
            if st.button("Predict Maintenance Requirement", type="primary"):
                try:
                    # Make prediction
                    prediction_proba = model.predict_proba(input_df)[0]
                    prediction = [1 if prediction_proba[1] >= 0.5 else 0]
                    
                    # Create columns for results
                    result_col1, result_col2 = st.columns([1, 1])
                    
                    with result_col1:
                        # Display prediction
                        if prediction[0] == 1:
                            st.markdown(f"""
                            <div class="prediction-box prediction-yes">
                                <h3>⚠️ Maintenance Required</h3>
                                <p>Confidence: {prediction_proba[1]:.2%}</p>
                                <p>The model predicts that this vehicle requires maintenance based on the provided data.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box prediction-no">
                                <h3>✅ No Maintenance Required</h3>
                                <p>Confidence: {prediction_proba[0]:.2%}</p>
                                <p>The model predicts that this vehicle does not require maintenance at this time.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display gauge chart
                        st.plotly_chart(create_gauge_chart(prediction_proba[1]), use_container_width=True)
                    
                    with result_col2:
                        # Display feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            # Get feature importance
                            feature_importance = model.feature_importances_
                            
                            # Create a DataFrame for visualization
                            importance_df = pd.DataFrame({
                                'Feature': columns,
                                'Importance': feature_importance
                            }).sort_values('Importance', ascending=False)
                            
                            # Display radar chart
                            st.plotly_chart(create_radar_chart(input_df, feature_importance), use_container_width=True)
                            
                            # Display top 3 factors
                            st.markdown("### Top Factors for This Prediction")
                            top_factors = importance_df.head(3)
                            for i, row in top_factors.iterrows():
                                feature = row['Feature']
                                importance = row['Importance']
                                value = input_df[feature].values[0]
                                st.write(f"**{feature}**: {value} (Importance: {importance:.4f})")
                    
                    # Display full feature importance chart
                    st.markdown("### Feature Importance")
                    st.write("This chart shows which factors most influenced the prediction:")
                    
                    # Plot feature importance
                    fig = px.bar(
                        importance_df.head(10), 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        title='Top 10 Features by Importance'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        
        with tab2:
            st.markdown("### Batch Prediction")
            st.write("Upload a CSV file with multiple vehicles to get predictions for your entire fleet.")
            
            # File uploader
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    batch_data = pd.read_csv(uploaded_file)
                    
                    # Show the uploaded data
                    st.write("Uploaded data:")
                    st.dataframe(batch_data)
                    
                    # Check if required columns are present
                    missing_cols = [col for col in columns if col not in [c.lower() for c in batch_data.columns]]
                    
                    if missing_cols:
                        st.warning(f"Missing columns in your data: {', '.join(missing_cols)}")
                        st.info("Please ensure your CSV has all required columns.")
                    else:
                        # Prepare data for prediction
                        # Convert column names to lowercase
                        batch_data.columns = [col.lower() for col in batch_data.columns]
                        
                        # Select only the columns needed for prediction
                        pred_data = batch_data[columns].copy()
                        
                        # Make predictions
                        if st.button("Run Batch Prediction", key="batch_predict"):
                            # Get predictions
                            batch_proba = model.predict_proba(pred_data)
                            batch_pred = model.predict(pred_data)
                            
                            # Add predictions to the dataframe
                            result_df = batch_data.copy()
                            result_df['Maintenance_Required'] = batch_pred
                            result_df['Maintenance_Probability'] = batch_proba[:, 1]
                            
                            # Display results
                            st.success(f"Processed {len(result_df)} vehicles")
                            
                            # Summary statistics
                            maintenance_count = result_df['Maintenance_Required'].sum()
                            st.metric("Vehicles Requiring Maintenance", maintenance_count, f"{maintenance_count/len(result_df):.1%} of fleet")
                            
                            # Display results table
                            st.write("Prediction Results:")
                            st.dataframe(result_df)
                            
                            # Visualization of results
                            fig = px.histogram(
                                result_df, 
                                x='Maintenance_Probability',
                                color='Maintenance_Required',
                                marginal='box',
                                title='Distribution of Maintenance Probability',
                                labels={'Maintenance_Probability': 'Probability of Maintenance Required', 'count': 'Number of Vehicles'},
                                color_discrete_map={0: 'green', 1: 'red'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download button for results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="fleet_maintenance_predictions.csv",
                                mime="text/csv",
                            )
                
                except Exception as e:
                    st.error(f"Error processing batch prediction: {e}")
            
            # Sample template
            st.markdown("### Need a template?")
            st.write("Download a sample CSV template to get started:")
            
            # Create a sample template
            template_data = pd.DataFrame({
                'Mileage_km': [50000, 75000],
                'Fuel_Consumption_L_per_100km': [8.5, 9.2],
                'Engine_Hours': [1200, 1800],
                'Number_of_Trips': [500, 750],
                'Engine_Temperature_C': [90, 95],
                'Tire_Pressure_PSI': [32.5, 31.0],
                'Battery_Voltage_V': [12.5, 12.2],
                'Oil_Level_L': [4.5, 4.2],
                'Vibration_Units': [2.5, 3.1],
                'Noise_Level_dB': [75, 82],
                'Breakdown_Incidents': [1, 2],
                'Part_Replacements': [2, 4],
                'Service_History': [3, 5]
            })
            
            csv = template_data.to_csv(index=False)
            st.download_button(
                label="Download Template CSV",
                data=csv,
                file_name="fleet_maintenance_template.csv",
                mime="text/csv",
            )
        
        with tab3:
            st.markdown("### What-If Analysis")
            st.write("Explore how changing vehicle parameters affects maintenance predictions.")
            
            # Base vehicle parameters
            st.markdown("#### Base Vehicle Parameters")
            base_params = {}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                base_params["mileage_km"] = st.number_input("Base Mileage (km)", min_value=0, value=50000, key="base_mileage")
                base_params["engine_hours"] = st.number_input("Base Engine Hours", min_value=0, value=1200, key="base_engine_hours")
                base_params["number_of_trips"] = st.number_input("Base Number of Trips", min_value=0, value=500, key="base_trips")
            
            with col2:
                base_params["fuel_consumption_l_per_100km"] = st.number_input("Base Fuel Consumption (L/100km)", min_value=0.0, value=8.5, step=0.1, key="base_fuel")
                base_params["engine_temperature_c"] = st.number_input("Base Engine Temperature (°C)", min_value=0, value=90, key="base_temp")
                base_params["tire_pressure_psi"] = st.number_input("Base Tire Pressure (PSI)", min_value=0.0, value=32.5, step=0.1, key="base_tire")
            
            with col3:
                base_params["battery_voltage_v"] = st.number_input("Base Battery Voltage (V)", min_value=0.0, value=12.5, step=0.1, key="base_battery")
                base_params["oil_level_l"] = st.number_input("Base Oil Level (L)", min_value=0.0, value=4.5, step=0.1, key="base_oil")
                base_params["vibration_units"] = st.number_input("Base Vibration Units", min_value=0.0, value=2.5, step=0.1, key="base_vibration")
            
            # Additional parameters
            st.markdown("#### Additional Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                base_params["noise_level_db"] = st.number_input("Base Noise Level (dB)", min_value=0, value=75, key="base_noise")
            
            with col2:
                base_params["breakdown_incidents"] = st.number_input("Base Breakdown Incidents", min_value=0, value=1, key="base_breakdown")
            
            with col3:
                base_params["part_replacements"] = st.number_input("Base Part Replacements", min_value=0, value=2, key="base_parts")
                base_params["service_history"] = st.number_input("Base Service History", min_value=0, value=3, key="base_service")
            
            # Parameter to vary
            st.markdown("#### What-If Parameter")
            col1, col2 = st.columns(2)
            
            with col1:
                vary_param = st.selectbox(
                    "Parameter to vary",
                    options=[
                        "mileage_km", "fuel_consumption_l_per_100km", "engine_hours", 
                        "number_of_trips", "engine_temperature_c", "tire_pressure_psi",
                        "battery_voltage_v", "oil_level_l", "vibration_units", 
                        "noise_level_db", "breakdown_incidents", "part_replacements", 
                        "service_history"
                    ],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            
            with col2:
                # Set range based on parameter
                if vary_param == "mileage_km":
                    min_val, max_val, step_val = 0, 150000, 10000
                elif vary_param == "fuel_consumption_l_per_100km":
                    min_val, max_val, step_val = 5.0, 15.0, 0.5
                elif vary_param == "engine_hours":
                    min_val, max_val, step_val = 0, 5000, 500
                elif vary_param == "number_of_trips":
                    min_val, max_val, step_val = 0, 2000, 200
                elif vary_param == "engine_temperature_c":
                    min_val, max_val, step_val = 70, 120, 5
                elif vary_param == "tire_pressure_psi":
                    min_val, max_val, step_val = 25.0, 40.0, 1.0
                elif vary_param == "battery_voltage_v":
                    min_val, max_val, step_val = 10.0, 14.0, 0.2
                elif vary_param == "oil_level_l":
                    min_val, max_val, step_val = 3.0, 6.0, 0.2
                elif vary_param == "vibration_units":
                    min_val, max_val, step_val = 0.0, 10.0, 0.5
                elif vary_param == "noise_level_db":
                    min_val, max_val, step_val = 60, 100, 5
                elif vary_param == "breakdown_incidents":
                    min_val, max_val, step_val = 0, 10, 1
                elif vary_param == "part_replacements":
                    min_val, max_val, step_val = 0, 20, 2
                elif vary_param == "service_history":
                    min_val, max_val, step_val = 0, 20, 2
                
                param_range = st.slider(
                    f"Range of {vary_param.replace('_', ' ').title()} values",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=(float(min_val), float(max_val)),
                    step=float(step_val)
                )
            
            # Number of steps
            num_steps = st.slider("Number of steps", min_value=5, max_value=50, value=20)
            
            # Run analysis button
            if st.button("Run What-If Analysis", type="primary"):
                # Generate parameter values
                param_values = np.linspace(param_range[0], param_range[1], num_steps)
                
                # Create dataframes for each parameter value
                results = []
                
                for val in param_values:
                    # Create a copy of base parameters
                    params = base_params.copy()
                    # Update the parameter to vary
                    params[vary_param] = val
                    
                    # Create dataframe
                    df = pd.DataFrame([params])
                    
                    # Ensure all columns are present
                    for col in columns:
                        if col not in df.columns:
                            df[col] = 0
                    
                    # Reorder columns
                    df = df[columns]
                    
                    # Make prediction
                    prob = model.predict_proba(df)[0][1]
                    pred = 1 if prob >= 0.5 else 0
                    
                    # Store results
                    results.append({
                        'Parameter_Value': val,
                        'Maintenance_Probability': prob,
                        'Maintenance_Required': pred
                    })
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Plot results
                fig = px.line(
                    results_df, 
                    x='Parameter_Value', 
                    y='Maintenance_Probability',
                    title=f'Effect of {vary_param.replace("_", " ").title()} on Maintenance Probability',
                    labels={
                        'Parameter_Value': vary_param.replace("_", " ").title(),
                        'Maintenance_Probability': 'Probability of Maintenance Required'
                    }
                )
                
                # Add threshold line
                fig.add_hline(
                    y=0.5, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Decision Threshold (0.5)",
                    annotation_position="bottom right"
                )
                
                # Add markers for maintenance required
                threshold_crossed = results_df[results_df['Maintenance_Required'] == 1]
                if not threshold_crossed.empty:
                    fig.add_vline(
                        x=threshold_crossed['Parameter_Value'].min(),
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="Maintenance Required Above This Value",
                        annotation_position="top left"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results table
                st.write("Detailed Results:")
                st.dataframe(results_df)
                
                # Find the threshold value where maintenance becomes required
                if 1 in results_df['Maintenance_Required'].values:
                    threshold_value = results_df[results_df['Maintenance_Required'] == 1]['Parameter_Value'].min()
                    st.info(f"Maintenance becomes required when {vary_param.replace('_', ' ').title()} reaches {threshold_value:.2f}")
    
    elif page == "Data Explorer":
        st.markdown('<p class="main-header">Fleet Data Explorer</p>', unsafe_allow_html=True)
        
        if sample_data is not None:
            # Display basic info about the dataset
            st.markdown("### Dataset Overview")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Summary", "📋 Data Table", "📈 Visualizations"])
            
            with tab1:
                # Display basic statistics
                st.write("Basic Statistics:")
                st.dataframe(sample_data.describe())
                
                # Display info about the dataset
                st.write("Dataset Information:")
                buffer = io.StringIO()
                sample_data.info(buf=buffer)
                st.text(buffer.getvalue())
                
                # Display maintenance distribution
                if 'MaintenanceRequired' in sample_data.columns:
                    st.write("Maintenance Distribution:")
                    maintenance_counts = sample_data['MaintenanceRequired'].value_counts()
                    fig = px.pie(
                        values=maintenance_counts.values,
                        names=maintenance_counts.index.map({0: 'No Maintenance', 1: 'Maintenance Required'}),
                        title='Maintenance Distribution',
                        color_discrete_sequence=['green', 'red']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Display the dataset
                st.write("Dataset:")
                st.dataframe(sample_data)
                
                # Add search functionality
                search_term = st.text_input("Search in dataset:")
                if search_term:
                    filtered_data = sample_data[sample_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
                    st.write(f"Found {len(filtered_data)} matching rows:")
                    st.dataframe(filtered_data)
            
            with tab3:
                st.write("Data Visualizations:")
                
                # Select columns for visualization
                numeric_cols = sample_data.select_dtypes(include=['number']).columns.tolist()
                
                # Correlation heatmap
                st.subheader("Correlation Heatmap")
                corr = sample_data[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature distributions
                st.subheader("Feature Distributions")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("X-axis", options=numeric_cols, index=0)
                
                with col2:
                    plot_type = st.selectbox(
                        "Plot Type", 
                        options=["Histogram", "Box Plot", "Violin Plot"],
                        index=0
                    )
                
                if 'MaintenanceRequired' in sample_data.columns:
                    color_by = st.checkbox("Color by Maintenance Required", value=True)
                    color_col = 'MaintenanceRequired' if color_by else None
                else:
                    color_col = None
                
                if plot_type == "Histogram":
                    fig = px.histogram(
                        sample_data, 
                        x=x_axis,
                        color=color_col,
                        marginal="box",
                        title=f"Distribution of {x_axis}",
                        color_discrete_map={0: 'green', 1: 'red'}
                    )
                elif plot_type == "Box Plot":
                    fig = px.box(
                        sample_data,
                        x=color_col if color_col else None,
                        y=x_axis,
                        color=color_col,
                        title=f"Box Plot of {x_axis}",
                        color_discrete_map={0: 'green', 1: 'red'}
                    )
                else:  # Violin Plot
                    fig = px.violin(
                        sample_data,
                        x=color_col if color_col else None,
                        y=x_axis,
                        color=color_col,
                        box=True,
                        title=f"Violin Plot of {x_axis}",
                        color_discrete_map={0: 'green', 1: 'red'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot
                st.subheader("Feature Relationships")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis_scatter = st.selectbox("X-axis", options=numeric_cols, index=0, key="scatter_x")
                
                with col2:
                    y_axis_scatter = st.selectbox("Y-axis", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
                
                with col3:
                    size_col = st.selectbox("Size (optional)", options=["None"] + numeric_cols, index=0)
                
                size = size_col if size_col != "None" else None
                
                fig = px.scatter(
                    sample_data,
                    x=x_axis_scatter,
                    y=y_axis_scatter,
                    color=color_col,
                    size=size,
                    title=f"{y_axis_scatter} vs {x_axis_scatter}",
                    color_discrete_map={0: 'green', 1: 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sample data not available. Please upload a dataset to explore.")
            
            # File uploader
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    data = pd.read_csv(uploaded_file)
                    
                    # Display the data
                    st.write("Uploaded data:")
                    st.dataframe(data)
                    
                    # Display basic statistics
                    st.write("Basic Statistics:")
                    st.dataframe(data.describe())
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    
    elif page == "About":
        st.markdown('<p class="main-header">About Fleet Maintenance Predictor</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>Fleet Maintenance Prediction System</h3>
            <p>This application uses machine learning to predict whether a vehicle requires maintenance based on various parameters.</p>
            <p>The model was trained using XGBoost on historical fleet maintenance data.</p>
            
            <h3>How to Use</h3>
            <ol>
                <li><strong>Prediction Dashboard:</strong> Enter vehicle details to get maintenance predictions</li>
                <li><strong>Batch Prediction:</strong> Upload a CSV file with multiple vehicles to get predictions for your entire fleet</li>
                <li><strong>What-If Analysis:</strong> Explore how changing vehicle parameters affects maintenance predictions</li>
                <li><strong>Data Explorer:</strong> Analyze and visualize your fleet data</li>
            </ol>
            
            <h3>Input Parameters</h3>
            <ul>
                <li><strong>Mileage (km)</strong>: Total distance traveled by the vehicle</li>
                <li><strong>Fuel Consumption (L/100km)</strong>: Average fuel consumption</li>
                <li><strong>Engine Hours</strong>: Total hours the engine has been running</li>
                <li><strong>Number of Trips</strong>: Total number of trips made</li>
                <li><strong>Engine Temperature (°C)</strong>: Average engine temperature</li>
                <li><strong>Tire Pressure (PSI)</strong>: Average tire pressure</li>
                <li><strong>Battery Voltage (V)</strong>: Battery voltage</li>
                <li><strong>Oil Level (L)</strong>: Engine oil level</li>
                <li><strong>Vibration Units</strong>: Measured vibration level</li>
                <li><strong>Noise Level (dB)</strong>: Measured noise level</li>
                <li><strong>Breakdown Incidents</strong>: Number of previous breakdowns</li>
                <li><strong>Part Replacements</strong>: Number of parts replaced</li>
                <li><strong>Service History</strong>: Number of previous services</li>
            </ul>
            
            <h3>Technical Details</h3>
            <p>The prediction model is built using XGBoost, a powerful gradient boosting algorithm that excels at classification tasks.</p>
            <p>Key features of the model:</p>
            <ul>
                <li>Trained on historical fleet maintenance data</li>
                <li>Optimized hyperparameters for best performance</li>
                <li>Feature importance analysis to understand key factors</li>
                <li>Probability-based predictions for maintenance requirements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical notes section
        st.markdown("### Technical Notes")
        st.markdown("""
        <div class="info-box">
            <h4>Model Deployment</h4>
            <p>This Streamlit application loads a pre-trained XGBoost model saved using pickle. For production use, consider:</p>
            
            <ul>
                <li>Retraining the model periodically with new data</li>
                <li>Monitoring model performance over time</li>
                <li>Implementing proper authentication for sensitive data</li>
                <li>Setting up automated data pipelines for batch predictions</li>
            </ul>
            
            <h4>Data Requirements</h4>
            <p>For accurate predictions, ensure your input data:</p>
            <ul>
                <li>Contains all required features</li>
                <li>Uses the same units as the training data</li>
                <li>Is preprocessed in the same way as the training data</li>
            </ul>
            
            <h4>Customization</h4>
            <p>This application can be customized to include:</p>
            <ul>
                <li>Integration with fleet management systems</li>
                <li>Automated alerts for vehicles requiring maintenance</li>
                <li>Historical tracking of predictions and actual maintenance</li>
                <li>Cost analysis based on predicted maintenance needs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()