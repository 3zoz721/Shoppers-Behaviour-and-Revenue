import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import warnings
warnings.filterwarnings('ignore')

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

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models"""
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    return results, trained_models

def create_ann_model(input_dim):
    """Create ANN model"""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main App
def main():
    st.markdown('<h1 class="main-header">🛒 Shoppers Behavior & Revenue Analytics</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["📊 Dashboard", "🔍 Data Exploration", "🤖 Model Training", "🎯 Predictions"])
    
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
    
    # Model Training Page
    elif page == "🤖 Model Training":
        st.header("Model Training & Evaluation")
        
        # Preprocessing
        df_encoded, scaler = preprocess_data(df)
        
        # Feature selection
        X = df_encoded.drop('Revenue', axis=1)
        y = df_encoded['Revenue']
        
        # Handle class imbalance
        if st.checkbox("Apply SMOTE for class imbalance"):
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            st.success("SMOTE applied successfully!")
        
        # Train-test split
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                           random_state=42, stratify=y)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Train traditional ML models
                results, trained_models = train_models(X_train, X_test, y_train, y_test)
                
                # Train ANN
                ann_model = create_ann_model(X_train.shape[1])
                history = ann_model.fit(X_train, y_train, validation_split=0.2, 
                                       epochs=30, batch_size=32, verbose=0)
                
                # ANN predictions
                ann_pred = ann_model.predict(X_test)
                ann_pred_binary = (ann_pred > 0.5).astype(int)
                ann_accuracy = accuracy_score(y_test, ann_pred_binary)
                
                results['ANN'] = {'accuracy': ann_accuracy, 'predictions': ann_pred_binary}
            
            # Display results
            st.subheader("Model Performance")
            
            # Create performance comparison
            model_names = list(results.keys())
            accuracies = [results[model]['accuracy'] for model in model_names]
            
            fig = px.bar(x=model_names, y=accuracies, 
                        title="Model Accuracy Comparison",
                        labels={'x': 'Model', 'y': 'Accuracy'})
            fig.update_traces(text=[f'{acc:.3f}' for acc in accuracies], textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            st.success(f"🏆 Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.3f})")
            
            # Detailed results
            st.subheader("Detailed Results")
            for model_name, result in results.items():
                with st.expander(f"{model_name} Results"):
                    st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, result['predictions'])
                    fig = px.imshow(cm, text_auto=True, aspect="auto",
                                   title=f"Confusion Matrix - {model_name}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    report = classification_report(y_test, result['predictions'], output_dict=True)
                    st.write("**Classification Report:**")
                    st.json(report)
            
            # Store models in session state
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.feature_names = X.columns.tolist()
    
    # Predictions Page
    elif page == "🎯 Predictions":
        st.header("Make Predictions")
        
        if 'trained_models' not in st.session_state:
            st.warning("Please train the models first in the Model Training page.")
            return
        
        st.subheader("Input Features")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            administrative = st.number_input("Administrative Pages", min_value=0, value=2)
            admin_duration = st.number_input("Administrative Duration", min_value=0.0, value=80.0)
            informational = st.number_input("Informational Pages", min_value=0, value=0)
            info_duration = st.number_input("Informational Duration", min_value=0.0, value=0.0)
            product_related = st.number_input("Product Related Pages", min_value=0, value=32)
            product_duration = st.number_input("Product Related Duration", min_value=0.0, value=1200.0)
        
        with col2:
            bounce_rates = st.slider("Bounce Rates", 0.0, 0.2, 0.02)
            exit_rates = st.slider("Exit Rates", 0.0, 0.2, 0.04)
            page_values = st.number_input("Page Values", min_value=0.0, value=5.0)
            special_day = st.slider("Special Day", 0.0, 1.0, 0.0)
            operating_systems = st.selectbox("Operating Systems", range(1, 9), index=1)
            browser = st.selectbox("Browser", range(1, 14), index=1)
        
        with col3:
            region = st.selectbox("Region", range(1, 10), index=2)
            traffic_type = st.selectbox("Traffic Type", range(1, 21), index=1)
            visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])
            weekend = st.checkbox("Weekend")
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        
        if st.button("Make Prediction"):
            # Prepare input data
            input_data = {
                'Administrative': administrative,
                'Administrative_Duration': admin_duration,
                'Informational': informational,
                'Informational_Duration': info_duration,
                'ProductRelated': product_related,
                'ProductRelated_Duration': product_duration,
                'BounceRates': bounce_rates,
                'ExitRates': exit_rates,
                'PageValues': page_values,
                'SpecialDay': special_day,
                'OperatingSystems': operating_systems,
                'Browser': browser,
                'Region': region,
                'TrafficType': traffic_type,
                'VisitorType': visitor_type,
                'Weekend': weekend,
                'Month': month
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables (simplified for demo)
            # In production, you'd want to use the same encoder used during training
            input_encoded = pd.get_dummies(input_df, columns=['Month', 'VisitorType', 'Weekend'], 
                                          drop_first=True, dtype=int)
            
            # Ensure all columns are present (pad with zeros if missing)
            for col in st.session_state.feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training data
            input_encoded = input_encoded.reindex(columns=st.session_state.feature_names, fill_value=0)
            
            # Scale numerical features
            cols_to_scale = ['Administrative_Duration', 'Informational_Duration', 
                           'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
            input_encoded[cols_to_scale] = st.session_state.scaler.transform(input_encoded[cols_to_scale])
            
            # Make predictions
            st.subheader("Prediction Results")
            
            for model_name, model in st.session_state.trained_models.items():
                prediction = model.predict(input_encoded)[0]
                probability = model.predict_proba(input_encoded)[0] if hasattr(model, 'predict_proba') else None
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{model_name}:**")
                    if prediction == 1:
                        st.success("✅ Will Purchase")
                    else:
                        st.error("❌ Won't Purchase")
                
                with col2:
                    if probability is not None:
                        st.write(f"Confidence: {max(probability):.2%}")
                        st.progress(max(probability))

if __name__ == "__main__":

    main()

