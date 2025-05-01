import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import google.generativeai as genai
from sklearn.linear_model import LinearRegression

# Set up page configuration
st.set_page_config(
    page_title="Cloud Service Cost Analyzer",
    page_icon="☁️",
    layout="wide"
)

# Global Gemini API configuration
API_KEY = "AIzaSyCnOxwQ7LUJy56rpH4tHKmKYqGZoabJ8rI"  
GEMINI_MODEL = "gemini-2.0-flash-lite"

# Configure Gemini globally
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# Function to get cloud service recommendations from Gemini
def get_service_recommendations(prompt, cloud_provider, region, scale):
    try:
        query = f"""
        I need detailed cloud service recommendations for a project with the following details:
        - User prompt: {prompt}
        - Cloud provider: {cloud_provider}
        - Region: {region}
        - Scale: {scale} ({get_scale_description(scale)})
        
        Please provide a detailed list of recommended services that would be needed for this use case.
        For each service, include:
        1. Service name
        2. Brief description of its purpose
        3. Typical usage metric (requests, GB, CPU hours, etc.)
        4. Estimated price per unit in USD
        5. Estimated quantity needed for the specified scale
        
        Format your response as a structured list that can be easily parsed.
        """
        
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"

# Function to extract JSON from text
def extract_json_from_text(text):
    # Try to find JSON in the text using regex
    json_pattern = r'\[\s*\{.*\}\s*\]'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# Function to process service recommendations text into structured data
def process_recommendations(recommendations_text):
    try:
        query = f"""
        Parse the following cloud service recommendations into a structured format:
        
        {recommendations_text}
        
        Return a JSON array where each item has these properties:
        - service_name: The name of the service
        - description: Brief description of the service
        - metric: The usage metric (requests, GB, hours, etc.)
        - price_per_unit: Estimated price per unit in USD (numeric value only)
        - quantity: Estimated quantity needed (numeric value only)
        - daily_cost: Estimated daily cost in USD (price_per_unit × quantity, numeric value only)
        
        Format as valid JSON array only. No markdown formatting, no backticks, just the pure JSON.
        """
        
        response = model.generate_content(query)
        response_text = response.text
        
        # Try to extract JSON
        json_text = extract_json_from_text(response_text)
        if not json_text:
            # Try to remove code blocks if present
            if "```json" in response_text:
                response_text = response_text.replace("```json", "").replace("```", "")
            elif "```" in response_text:
                response_text = response_text.split("```")[1]
            json_text = response_text.strip()
        
        # For debugging (internal only)
        debug_info = json_text
        
        # Parse JSON with error handling
        try:
            # Try direct parsing first
            services_data = json.loads(json_text)
            
            # Verify we got a list/array
            if not isinstance(services_data, list):
                st.error("Invalid response format. Expected JSON array.")
                return []
                
            # Fix numerical values
            for service in services_data:
                for key in ['price_per_unit', 'quantity', 'daily_cost']:
                    if key in service and isinstance(service[key], str):
                        try:
                            service[key] = float(service[key])
                        except ValueError:
                            service[key] = 0.0
                
            return services_data
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {str(e)}")
            
            # Try creating services manually
            st.warning("Attempting to create services manually from response...")
            
            # Attempt to manually create a structured format
            services = []
            lines = response_text.strip().split("\n")
            current_service = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith("- service_name:") or line.startswith("\"service_name\":"):
                    if current_service and 'service_name' in current_service:
                        services.append(current_service)
                    current_service = {'service_name': line.split(":", 1)[1].strip().strip(',"')}
                elif ": " in line:
                    key, value = line.split(":", 1)
                    key = key.strip().strip('-"')
                    value = value.strip().strip(',"')
                    if key in ['price_per_unit', 'quantity', 'daily_cost'] and value:
                        try:
                            current_service[key] = float(value)
                        except ValueError:
                            current_service[key] = 0.0
                    elif key in ['service_name', 'description', 'metric']:
                        current_service[key] = value
            
            if current_service and 'service_name' in current_service:
                services.append(current_service)
                
            if services:
                st.success(f"Successfully created {len(services)} services manually")
                return services
            
            return []
    except Exception as e:
        st.error(f"Error processing recommendations: {str(e)}")
        return []

# Function to manually generate sample data if everything else fails
def generate_sample_data(cloud_provider, scale):
    if cloud_provider == "AWS":
        services = [
            {
                "service_name": "Amazon S3",
                "description": "Object storage for storing data and assets",
                "metric": "GB-month",
                "price_per_unit": 0.023,
                "quantity": 10,
                "daily_cost": 0.023 * 10 / 30
            },
            {
                "service_name": "Amazon EC2",
                "description": "Virtual server instance",
                "metric": "hours",
                "price_per_unit": 0.10,
                "quantity": 24,
                "daily_cost": 0.10 * 24
            },
            {
                "service_name": "Amazon RDS",
                "description": "Managed relational database service",
                "metric": "hours",
                "price_per_unit": 0.17,
                "quantity": 24,
                "daily_cost": 0.17 * 24
            },
            {
                "service_name": "AWS Lambda",
                "description": "Serverless computing service",
                "metric": "invocations",
                "price_per_unit": 0.0000002,
                "quantity": 10000,
                "daily_cost": 0.0000002 * 10000
            },
            {
                "service_name": "Amazon CloudFront",
                "description": "Content delivery network",
                "metric": "GB transferred",
                "price_per_unit": 0.085,
                "quantity": 5,
                "daily_cost": 0.085 * 5
            }
        ]
    elif cloud_provider == "GCP":
        services = [
            {
                "service_name": "Cloud Storage",
                "description": "Object storage for storing data and assets",
                "metric": "GB-month",
                "price_per_unit": 0.020,
                "quantity": 10,
                "daily_cost": 0.020 * 10 / 30
            },
            {
                "service_name": "Compute Engine",
                "description": "Virtual machine instances",
                "metric": "hours",
                "price_per_unit": 0.09,
                "quantity": 24,
                "daily_cost": 0.09 * 24
            },
            {
                "service_name": "Cloud SQL",
                "description": "Managed relational database service",
                "metric": "hours",
                "price_per_unit": 0.16,
                "quantity": 24,
                "daily_cost": 0.16 * 24
            },
            {
                "service_name": "Cloud Functions",
                "description": "Serverless computing platform",
                "metric": "invocations",
                "price_per_unit": 0.0000004,
                "quantity": 10000,
                "daily_cost": 0.0000004 * 10000
            },
            {
                "service_name": "BigQuery",
                "description": "Data warehouse and analytics",
                "metric": "GB processed",
                "price_per_unit": 0.005,
                "quantity": 20,
                "daily_cost": 0.005 * 20
            }
        ]
    else:  # Azure
        services = [
            {
                "service_name": "Blob Storage",
                "description": "Object storage for storing data and assets",
                "metric": "GB-month",
                "price_per_unit": 0.018,
                "quantity": 10,
                "daily_cost": 0.018 * 10 / 30
            },
            {
                "service_name": "Virtual Machines",
                "description": "Virtual machine instances",
                "metric": "hours",
                "price_per_unit": 0.08,
                "quantity": 24,
                "daily_cost": 0.08 * 24
            },
            {
                "service_name": "Azure SQL",
                "description": "Managed relational database service",
                "metric": "hours",
                "price_per_unit": 0.15,
                "quantity": 24,
                "daily_cost": 0.15 * 24
            },
            {
                "service_name": "Azure Functions",
                "description": "Serverless compute service",
                "metric": "executions",
                "price_per_unit": 0.0000002,
                "quantity": 10000,
                "daily_cost": 0.0000002 * 10000
            },
            {
                "service_name": "Azure Data Factory",
                "description": "Data integration service",
                "metric": "hours",
                "price_per_unit": 0.001,
                "quantity": 24,
                "daily_cost": 0.001 * 24
            }
        ]
    
    # Scale the quantities based on the selected scale
    scale_factor = 1
    if scale == "Medium":
        scale_factor = 10
    elif scale == "Large":
        scale_factor = 100
    
    for service in services:
        service["quantity"] *= scale_factor
        service["daily_cost"] *= scale_factor
    
    return services

# Function to get scale description
def get_scale_description(scale):
    if scale == "Small":
        return "approximately 100 requests per day"
    elif scale == "Medium":
        return "approximately 1,000 requests per day"
    elif scale == "Large":
        return "approximately 10,000 requests per day"
    return ""

# Function to generate cost projections
def generate_cost_projections(services_data, scale):
    # Extract the daily cost from services data
    daily_cost = sum(service['daily_cost'] for service in services_data)
    
    # Set scale factors based on user selection
    if scale == "Small":
        requests_per_day = 100
    elif scale == "Medium":
        requests_per_day = 1000
    else:  # Large
        requests_per_day = 10000
    
    # Create training data for linear regression
    X = np.array([[100], [1000], [10000]])
    
    # Estimate costs based on the scale factor
    # This is a simplified model; in reality, costs might not scale linearly
    small_cost = daily_cost * (100 / requests_per_day)
    medium_cost = daily_cost * (1000 / requests_per_day)
    large_cost = daily_cost * (10000 / requests_per_day)
    
    y = np.array([small_cost, medium_cost, large_cost])
    
    # Train linear regression model
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    
    # Generate projections
    projections = {
        'Daily': daily_cost,
        'Weekly': daily_cost * 7,
        'Monthly': daily_cost * 30
    }
    
    return projections, reg_model

# Function to create service cost visualization with Plotly
def create_service_cost_chart(services_data):
    df = pd.DataFrame(services_data)
    
    # Define a custom color scale
    colors = px.colors.qualitative.Vivid
    
    fig = px.bar(
        df, 
        x='service_name', 
        y='daily_cost',
        color='service_name',
        color_discrete_sequence=colors,
        labels={'daily_cost': 'Daily Cost (USD)', 'service_name': 'Service'},
        title='Daily Cost by Service'
    )
    
    # Customize the layout
    fig.update_layout(
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=500,
        xaxis_title='Cloud Services',
        yaxis_title='Daily Cost (USD)',
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)'
        )
    )
    
    # Format hover info to display service details
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Daily Cost: $%{y:.2f}<extra></extra>'
    )
    
    # Add dollar sign to y-axis tick labels
    fig.update_yaxes(tickprefix="$")
    
    return fig

# Function to create pie chart of service costs with Plotly
def create_cost_distribution_pie(services_data):
    df = pd.DataFrame(services_data)
    
    # Add percentage for label display
    total_cost = df['daily_cost'].sum()
    df['percentage'] = df['daily_cost'] / total_cost * 100
    
    # Create hover text with multiple lines of information
    hover_text = []
    for index, row in df.iterrows():
        hover_text.append(f"<b>{row['service_name']}</b><br>" +
                          f"Description: {row['description']}<br>" +
                          f"Metric: {row['metric']}<br>" +
                          f"Cost: ${row['daily_cost']:.2f} ({row['percentage']:.1f}%)")
    
    fig = px.pie(
        df, 
        values='daily_cost', 
        names='service_name',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title='Cost Distribution by Service',
        hover_data=['percentage']
    )
    
    # Customize hover info
    fig.update_traces(
        hovertemplate='%{customdata[0]:.1f}%<br>$%{value:.2f}<extra></extra>',
        textinfo='none'  
    )
    
    # Add custom hover labels
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=500
    )
    
    return fig

# Function to create time projection chart with Plotly
def create_time_projection_chart(projections):
    periods = list(projections.keys())
    costs = list(projections.values())
    
    fig = px.line(
        x=periods, 
        y=costs,
        markers=True,
        title='Cost Projections Over Time',
        labels={'x': 'Time Period', 'y': 'Projected Cost (USD)'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add data points as scatter
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=costs,
            mode='markers+text',
            marker=dict(
                color='#1f77b4',
                size=12,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"${cost:.2f}" for cost in costs],
            textposition="top center",
            showlegend=False
        )
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=500,
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        xaxis=dict(
            showgrid=False
        )
    )
    
    # Add dollar sign to y-axis tick labels
    fig.update_yaxes(tickprefix="$")
    
    return fig

# Function to create scaling comparison chart with Plotly
def create_scaling_comparison(model):
    # Generate predictions for different request volumes
    request_volumes = np.arange(100, 15000, 1000)
    predicted_costs = model.predict(request_volumes.reshape(-1, 1))
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Requests': request_volumes,
        'Cost': predicted_costs
    })
    
    fig = px.line(
        df, 
        x='Requests', 
        y='Cost',
        title='Cost Scaling by Request Volume',
        labels={'Requests': 'Number of Requests per Day', 'Cost': 'Estimated Daily Cost (USD)'},
        color_discrete_sequence=['#ff7f0e']
    )
    
    # Add markers at specific request volumes
    key_points = [100, 1000, 10000]
    key_costs = model.predict(np.array(key_points).reshape(-1, 1))
    
    fig.add_trace(
        go.Scatter(
            x=key_points,
            y=key_costs.flatten(),
            mode='markers+text',
            marker=dict(
                color='#ff7f0e',
                size=12,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[f"${cost:.2f}" for cost in key_costs],
            textposition="top center",
            showlegend=False
        )
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=500,
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    # Add dollar sign to y-axis tick labels
    fig.update_yaxes(tickprefix="$")
    
    # Improve hover information
    fig.update_traces(
        hovertemplate='Requests: %{x}<br>Daily Cost: $%{y:.2f}<extra></extra>'
    )
    
    return fig

# Function to create cumulative cost chart with Plotly
def create_cumulative_cost_chart(projections):
    # Calculate cumulative costs over 30 days
    daily_cost = projections['Daily']
    days = range(1, 31)
    cumulative_costs = [daily_cost * day for day in days]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Day': days,
        'Cumulative Cost': cumulative_costs
    })
    
    fig = px.area(
        df, 
        x='Day', 
        y='Cumulative Cost',
        title='Cumulative Cost Over 30 Days',
        labels={'Day': 'Days', 'Cumulative Cost': 'Cumulative Cost (USD)'},
        color_discrete_sequence=['#2ca02c']
    )
    
    # Add markers at key points
    key_points = [1, 7, 30]
    for point in key_points:
        if point <= 30:
            fig.add_annotation(
                x=point,
                y=daily_cost * point,
                text=f"Day {point}: ${daily_cost * point:.2f}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=0,
                ay=-40
            )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=500,
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        )
    )
    
    # Add dollar sign to y-axis tick labels
    fig.update_yaxes(tickprefix="$")
    
    # Improve hover information
    fig.update_traces(
        hovertemplate='Day: %{x}<br>Cumulative Cost: $%{y:.2f}<extra></extra>'
    )
    
    return fig

# Function to create provider comparison chart with Plotly
def create_provider_comparison_chart(comparison_data):
    providers = list(comparison_data.keys())
    monthly_costs = [comparison_data[provider]['Monthly'] for provider in providers]
    
    # Create color mapping for providers
    provider_colors = {
        'AWS': '#FF9900',  # AWS orange
        'GCP': '#4285F4',  # Google blue
        'Azure': '#008AD7'  # Azure blue
    }
    
    # Map colors to providers in the data
    colors = [provider_colors.get(provider, '#1f77b4') for provider in providers]
    
    fig = px.bar(
        x=providers, 
        y=monthly_costs,
        title='Monthly Cost Comparison by Cloud Provider',
        labels={'x': 'Cloud Provider', 'y': 'Monthly Cost (USD)'},
        color=providers,
        color_discrete_map={provider: color for provider, color in zip(providers, colors)}
    )
    
    # Add cost labels on top of bars
    fig.update_traces(
        texttemplate='$%{y:.2f}',
        textposition='outside'
    )
    
    # Customize layout
    fig.update_layout(
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=500,
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        showlegend=False
    )
    
    # Add dollar sign to y-axis tick labels
    fig.update_yaxes(tickprefix="$")
    
    # Improve hover information
    fig.update_traces(
        hovertemplate='%{x}<br>Monthly Cost: $%{y:.2f}<extra></extra>'
    )
    
    return fig

# Main app
def main():
    # App title and description
    st.title("☁️ Cloud Service Cost Analyzer")
    st.markdown("""
    📍Enter your project details below to get started.  
    📍 Also specify the type of cloud service provider in the drop-down.  
    📍 The region specifies a geographical area where a cloud provider's infrastructure, including data centers, is located.  
    📍 The choice in the varying of the region also determines the cost of the project.   
    """)
    
    # User input section
    st.header("Project Requirements")
    
    # Create columns for form inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Text area for project description
        prompt = st.text_area("Project Description", 
                              "Build a web application that processes customer data and generates reports.")
        
        # Cloud provider selection
        cloud_provider = st.selectbox("Cloud Provider", 
                                     ["AWS", "GCP", "Azure"])
    
    with col2:
        # Region selection based on cloud provider
        regions = {
            "AWS": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            "GCP": ["us-central1", "europe-west1", "asia-east1", "australia-southeast1"],
            "Azure": ["East US", "West Europe", "Southeast Asia", "Australia East"]
        }
        
        region = st.selectbox("Region", regions[cloud_provider])
        
        # Scale selection
        scale = st.radio("Project Scale", 
                        ["Small", "Medium", "Large"],
                        help="Small: ~100 requests/day, Medium: ~1k requests/day, Large: ~10k requests/day")
    
    # Button to generate recommendations
    if st.button("Analyze Cloud Services and Costs"):
        with st.spinner("Analyzing with Gemini AI... This may take a moment."):
            # Get recommendations
            recommendations_text = get_service_recommendations(prompt, cloud_provider, region, scale)
            
            # Process recommendations (hide raw response from UI)
            services_data = process_recommendations(recommendations_text)
            
            # If services data is empty, try generating sample data
            if not services_data or len(services_data) == 0:
                st.warning("Could not generate service recommendations. Using sample data instead.")
                services_data = generate_sample_data(cloud_provider, scale)
            
            # Generate cost projections
            projections, regression_model = generate_cost_projections(services_data, scale)
            
            # Display results
            st.header("Recommended Cloud Services")
            
            # Create and display service cost table
            df = pd.DataFrame(services_data)
            
            # Format numerical values in the DataFrame
            display_df = df[['service_name', 'description', 'metric', 'price_per_unit', 'quantity', 'daily_cost']].copy()
            display_df['price_per_unit'] = display_df['price_per_unit'].apply(lambda x: f"${x:.6f}")
            display_df['daily_cost'] = display_df['daily_cost'].apply(lambda x: f"${x:.2f}")
            
            # Display the table with improved formatting
            st.dataframe(
                display_df,
                column_config={
                    "service_name": "Service Name",
                    "description": "Description",
                    "metric": "Usage Metric",
                    "price_per_unit": "Price Per Unit",
                    "quantity": "Estimated Quantity",
                    "daily_cost": "Daily Cost"
                },
                hide_index=True
            )
            
            # Display cost projections
            st.header("Cost Projections")
            
            # Create columns for displaying projections
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Daily Cost", f"${projections['Daily']:.2f}")
            
            with col2:
                st.metric("Weekly Cost", f"${projections['Weekly']:.2f}")
            
            with col3:
                st.metric("Monthly Cost", f"${projections['Monthly']:.2f}")
            
            # Visualizations
            st.header("Visualizations")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Service Costs", "Cost Distribution", 
                "Time Projection", "Scaling Comparison", 
                "Cumulative Cost"
            ])
            
            with tab1:
                st.plotly_chart(create_service_cost_chart(services_data), use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_cost_distribution_pie(services_data), use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_time_projection_chart(projections), use_container_width=True)
            
            with tab4:
                st.plotly_chart(create_scaling_comparison(regression_model), use_container_width=True)
            
            with tab5:
                st.plotly_chart(create_cumulative_cost_chart(projections), use_container_width=True)
            
            # Store current results for comparison
            st.session_state[f"{cloud_provider}_data"] = {
                'services': services_data,
                'projections': projections
            }
            
            # Multi-cloud comparison button
            st.header("Multi-Cloud Comparison")
            if st.button("Compare All Cloud Providers"):
                with st.spinner("Generating multi-cloud comparison... This may take several minutes."):
                    comparison_data = {}
                    
                    # Check if we already have the current provider data
                    if f"{cloud_provider}_data" in st.session_state:
                        comparison_data[cloud_provider] = st.session_state[f"{cloud_provider}_data"]['projections']
                    
                    # Get data for other providers
                    providers = ["AWS", "GCP", "Azure"]
                    for provider in providers:
                        if provider != cloud_provider and f"{provider}_data" not in st.session_state:
                            # Get recommendations for this provider
                            provider_recs = get_service_recommendations(prompt, provider, regions[provider][0], scale)
                            provider_services = process_recommendations(provider_recs)
                            
                            # If provider services is empty, use sample data
                            if not provider_services or len(provider_services) == 0:
                                provider_services = generate_sample_data(provider, scale)
                            
                            provider_projections, _ = generate_cost_projections(provider_services, scale)
                            st.session_state[f"{provider}_data"] = {
                                'services': provider_services,
                                'projections': provider_projections
                            }
                            comparison_data[provider] = provider_projections
                        elif f"{provider}_data" in st.session_state:
                            comparison_data[provider] = st.session_state[f"{provider}_data"]['projections']
                    
                    # Display comparison
                    st.header("Multi-Cloud Cost Comparison")
                    
                    # Create comparison table
                    comparison_df = pd.DataFrame({
                        provider: {
                            'Daily': data['Daily'],
                            'Weekly': data['Weekly'],
                            'Monthly': data['Monthly']
                        } for provider, data in comparison_data.items()
                    })
                    
                    st.dataframe(comparison_df)
                    
                    # Display comparison chart
                    st.pyplot(create_provider_comparison_chart(comparison_data))
                    
                    # Recommendation based on cost
                    min_provider = min(comparison_data.items(), key=lambda x: x[1]['Monthly'])
                    st.success(f"Based on cost analysis, {min_provider[0]} appears to be the most cost-effective option for your project with an estimated monthly cost of ${min_provider[1]['Monthly']:.2f}.")

if __name__ == "__main__":
    main()