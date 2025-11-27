import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚗 Car Price Prediction System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Analysis", "🤖 Model Training", "🔮 Price Prediction"])

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('Vehicle.csv')
        
        # Handle missing values - numerical
        df['price'] = df['price'].fillna(df['price'].median())
        df['mileage'] = df['mileage'].fillna(df['mileage'].median())
        
        # Handle missing values - categorical
        categorical_cols = ['description', 'cylinders', 'fuel', 'transmission', 
                          'trim', 'body', 'doors', 'exterior_color', 'interior_color']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Handle engine column
        if 'engine' in df.columns:
            invalid_values = ['o', ';', 'c', 'der', 'ER', 'oBoost', 'o ', 'c ', 
                            'oZEV', 'c ZEV', 'o ZEV', 'ZEV', 'o Z', 'oZ', 
                            'd>\\n\\n    \\n    <dt>VIN', '<dt>VIN']
            df['engine'] = df['engine'].replace(invalid_values, np.nan)
            df['engine'] = df['engine'].apply(lambda x: np.nan if isinstance(x, str) 
                                             and '<dt>VIN' in x else x)
            df['engine'] = df['engine'].fillna(df['engine'].mode()[0])
        
        # Drop unnecessary columns
        cols_to_drop = ['description', 'VIN', 'image_url', 'id']
        df = df.drop([col for col in cols_to_drop if col in df.columns], 
                    axis=1, errors='ignore')
        
        # Create log-transformed price
        df['price_log'] = np.log1p(df['price'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_and_preprocess_data()

if df is not None:
    # Page 1: Data Analysis
    if page == "📊 Data Analysis":
        st.header("📊 Exploratory Data Analysis")
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Avg Price", f"${df['price'].mean():,.0f}")
        
        st.markdown("---")
        
        # Data preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Price distribution
        st.subheader("Price Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='price', nbins=50, 
                             title='Distribution of Car Prices',
                             labels={'price': 'Price ($)', 'count': 'Frequency'})
            fig.update_traces(marker_color='lightblue', marker_line_color='darkblue', 
                            marker_line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y='price', title='Price Distribution (Box Plot)',
                        labels={'price': 'Price ($)'})
            fig.update_traces(marker_color='lightcoral')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Correlation for numerical features
        st.subheader("Correlation Analysis")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Feature Correlation Heatmap",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Page 2: Model Training
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")
        
        # Prepare data
        df_model = pd.get_dummies(df.drop(['price', 'price_log'], axis=1), drop_first=True)
        df_model.columns = df_model.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_')
        
        X = df_model
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.info("💡 Training models on your dataset...")
        
        # Train models
        models = {}
        results = {}
        
        # Linear Regression
        with st.spinner("Training Linear Regression..."):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            models['Linear Regression'] = lr
            results['Linear Regression'] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'R²': r2_score(y_test, lr_pred)
            }
        
        # Random Forest
        with st.spinner("Training Random Forest..."):
            rf = RandomForestRegressor(random_state=42, n_estimators=100)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            models['Random Forest'] = rf
            results['Random Forest'] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'R²': r2_score(y_test, rf_pred)
            }
        
        # XGBoost
        with st.spinner("Training XGBoost..."):
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, 
                             max_depth=5, random_state=42)
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)
            models['XGBoost'] = xgb
            results['XGBoost'] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'R²': r2_score(y_test, xgb_pred)
            }
        
        # Display results
        st.success("✅ Models trained successfully!")
        
        st.subheader("Model Performance Comparison")
        
        # Create comparison dataframe
        results_df = pd.DataFrame(results).T
        results_df = results_df.reset_index().rename(columns={'index': 'Model'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Model', y='RMSE', 
                        title='RMSE Comparison (Lower is Better)',
                        color='Model')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df, x='Model', y='R²', 
                        title='R² Score Comparison (Higher is Better)',
                        color='Model')
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(results_df, use_container_width=True)
        
        # Cross-validation for best model
        st.subheader("Cross-Validation Analysis (XGBoost)")
        cv_scores = cross_val_score(xgb, X, y, cv=5, scoring='r2')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean R² Score", f"{cv_scores.mean():.4f}")
        with col2:
            st.metric("Std Deviation", f"{cv_scores.std():.4f}")
        with col3:
            st.metric("Best Fold Score", f"{cv_scores.max():.4f}")
        
        # Save best model
        st.session_state['best_model'] = xgb
        st.session_state['feature_columns'] = X.columns.tolist()
    
    # Page 3: Price Prediction
    elif page == "🔮 Price Prediction":
        st.header("🔮 Predict Car Price")
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Please train the models first from the 'Model Training' page!")
        else:
            st.info("Enter car details below to predict its price")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year = st.number_input("Year", min_value=1990, max_value=2024, value=2020)
                mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=50000)
                
            with col2:
                condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
                fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
                
            with col3:
                transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
                body = st.selectbox("Body Type", ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"])
            
            if st.button("🔮 Predict Price", type="primary"):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'year': [year],
                    'mileage': [mileage],
                    'condition': [condition],
                    'fuel': [fuel],
                    'transmission': [transmission],
                    'body': [body]
                })
                
                # Add dummy columns to match training data
                input_encoded = pd.get_dummies(input_data)
                
                # Align with training columns
                for col in st.session_state['feature_columns']:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                input_encoded = input_encoded[st.session_state['feature_columns']]
                
                # Predict
                prediction = st.session_state['best_model'].predict(input_encoded)[0]
                
                # Display result
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; 
                                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                  border-radius: 1rem; color: white;'>
                            <h2>Predicted Price</h2>
                            <h1 style='font-size: 3rem; margin: 1rem 0;'>${prediction:,.2f}</h1>
                            <p>Based on XGBoost Model</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence interval (simplified)
                st.markdown("---")
                st.markdown("### Price Range Estimate")
                lower_bound = prediction * 0.90
                upper_bound = prediction * 1.10
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lower Bound (10% less)", f"${lower_bound:,.2f}")
                with col2:
                    st.metric("Predicted Price", f"${prediction:,.2f}")
                with col3:
                    st.metric("Upper Bound (10% more)", f"${upper_bound:,.2f}")

else:
    st.error("❌ Could not load the dataset. Please ensure 'Vehicle.csv' is in the same directory.")