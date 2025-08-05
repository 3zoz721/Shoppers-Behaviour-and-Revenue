import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML Libraries
from sklearn.preprocessing import StandardScaler



# Page Configuration
st.set_page_config(
    page_title="Shoppers Behavior Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # You'll need to upload your CSV file
        df = pd.read_csv("Shoppers_Behaviour_and_Revenue.csv")
        
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        # Convert Revenue to binary
        df['Revenue'] = df['Revenue'].astype(int)
        
        return df
    except FileNotFoundError:
        st.error("Please upload the 'Shoppers_Behaviour_and_Revenue.csv' file to the same directory as this script.")
        return None

def preprocess_data(df):
    """Preprocess data for machine learning"""
    # Scale numerical features
    cols_to_scale = [
        'Administrative_Duration', 'Informational_Duration', 
        'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues'
    ]
    
    scaler = StandardScaler()
    df_processed = df.copy()
    df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df_processed, columns=['Month', 'VisitorType', 'Weekend'], 
                               drop_first=True, dtype=int)
    
    return df_encoded, scaler



# Main App
def main():
    st.markdown('<h1 class="main-header">🛒 Shoppers Behavior & Revenue Analytics</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["📊 Dashboard", "🔍 Data Exploration"])
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Dashboard Page
    if page == "📊 Dashboard":
        st.header("Executive Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Sessions</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            purchase_rate = df['Revenue'].mean() * 100
            st.markdown("""
            <div class="metric-card">
                <h3>Purchase Rate</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(purchase_rate), unsafe_allow_html=True)
        
        with col3:
            avg_duration = df['ProductRelated_Duration'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Session Duration</h3>
                <h2>{:.0f}s</h2>
            </div>
            """.format(avg_duration), unsafe_allow_html=True)
        
        with col4:
            bounce_rate = df['BounceRates'].mean() * 100
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Bounce Rate</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(bounce_rate), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue Distribution
            fig = px.pie(df, names='Revenue', title='Revenue Distribution',
                        labels={0: 'No Purchase', 1: 'Purchase'})
            fig.update_traces(labels=['No Purchase', 'Purchase'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Visitor Type Analysis
            visitor_revenue = df.groupby('VisitorType')['Revenue'].agg(['count', 'mean']).reset_index()
            fig = px.bar(visitor_revenue, x='VisitorType', y='mean', 
                        title='Purchase Rate by Visitor Type',
                        labels={'mean': 'Purchase Rate', 'VisitorType': 'Visitor Type'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Analysis
        monthly_data = df.groupby('Month')['Revenue'].agg(['count', 'sum', 'mean']).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=monthly_data['Month'], y=monthly_data['count'], name="Total Sessions"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_data['Month'], y=monthly_data['mean'], 
                      mode='lines+markers', name="Purchase Rate", line=dict(color='red')),
            secondary_y=True,
        )
        
        fig.update_layout(title_text="Monthly Sessions and Purchase Rate")
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Number of Sessions", secondary_y=False)
        fig.update_yaxes(title_text="Purchase Rate", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Exploration Page
    elif page == "🔍 Data Exploration":
        st.header("Data Exploration")
        
        # Dataset Overview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Duplicates:** {df.duplicated().sum()}")
        
        with col2:
            st.subheader("Data Types")
            st.write(df.dtypes.value_counts())
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Correlation Heatmap
        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Correlation Heatmap",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Analysis
        st.subheader("Feature Analysis")
        
        feature = st.selectbox("Select feature to analyze:", 
                              ['ProductRelated', 'ProductRelated_Duration', 'BounceRates', 
                               'ExitRates', 'PageValues', 'SpecialDay'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=feature, color='Revenue', nbins=30,
                             title=f'Distribution of {feature} by Revenue')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Revenue', y=feature,
                        title=f'{feature} by Revenue')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
