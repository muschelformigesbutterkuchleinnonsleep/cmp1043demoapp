import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import requests
from openai import OpenAI

client = OpenAI(api_key=st.secrets["sk-proj"])

def get_ai_analysis(prompt):
    try:
        response = client.completions.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

st.set_page_config(page_title='CMP1043 Demo App by Madeleine Ong', layout='wide')
st.title('Analysis the relationship between two variables')
st.markdown('Upload your CSV file for me to analyze relationships between variables')

#File upload
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        #Display dataset information
        st.subheader('Dataset Preview')
        st.dataframe(data.head())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Dataset Info')
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())

        with col2:
            st.subheader('Summary Statistics')
            st.write(data.describe())

        #I will let the user select columns for analysis
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if len(numeric_columns) < 2:
            st.error('Please upload a dataset with at least 2 numeric columns')
        else:
            st.subheader('Select Variables for Analysis')
            x_column = st.selectbox('Select X variable', numeric_columns, index=0)
            y_column = st.selectbox('Select Y variable', numeric_columns, index=min(1, len(numeric_columns)-1))

            #Data preprocessing
            st.subheader('Data Preprocessing')

            #I create a copy of the data with only selected columns
            analysis_data = data[[x_column, y_column]].copy()

            #Missing values handling
            missing_values = analysis_data.isnull().sum().sum()
            if missing_values > 0:
                st.write(f'Found {missing_values} missing value(s) in selected columns')
                missing_method = st.radio(
                    'Choose method to handle missing values:',
                    ('Drop rows with missing values', 'Fill missing values with mean')
                )

                if missing_method == 'Drop rows with missing values':
                    analysis_data = analysis_data.dropna()
                    st.write(f'Dropped {missing_values} rows with missing values')
                else:
                    analysis_data = analysis_data.fillna(analysis_data.mean())
                    st.write('Filled missing values with column means')

            #Outlier removal by using IQR
            remove_outliers = st.checkbox('Remove outliers using IQR method')
            if remove_outliers:
                original_rows = len(analysis_data)

                #Calculate IQR for each column
                Q1_x = analysis_data[x_column].quantile(0.25)
                Q3_x = analysis_data[x_column].quantile(0.75)
                IQR_x = Q3_x - Q1_x

                Q1_y = analysis_data[y_column].quantile(0.25)
                Q3_y = analysis_data[y_column].quantile(0.75)
                IQR_y = Q3_y - Q1_y

                #Filter outliers
                analysis_data = analysis_data[
                    (analysis_data[x_column] >= Q1_x - 1.5 * IQR_x) &
                    (analysis_data[x_column] <= Q3_x + 1.5 * IQR_x) &
                    (analysis_data[y_column] >= Q1_y - 1.5 * IQR_y) &
                    (analysis_data[y_column] <= Q3_y + 1.5 * IQR_y)
                ]

                remove_rows = original_rows - len(analysis_data)
                st.write(f'Removed {remove_rows} outliers by using IQR method')

            #Data normalization option
            normalize_data = st.checkbox('Normalize or you can call it scale data')
            X_scaled = analysis_data[x_column].values.reshape(-1, 1)
            y_scaled = analysis_data[y_column].values

            if normalize_data:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_scaled = scaler_X.fit_transform(X_scaled)
                y_scaled = scaler_y.fit_transform(y_scaled.reshape(-1, 1)).flatten()

                st.write('Data has been normalized by using StandardScaler')

            #Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )

            #Data visualization
            st.subheader('Data Visualization')

            fig, ax = plt.subplots(figsize=(10, 6))

            #Scatter plot here
            sns.scatterplot(x=analysis_data[x_column], y=analysis_data[y_column], ax=ax)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f'Scatter plot: {x_column} vs {y_column}')

            #Calculate and display statistics
            mean_x = analysis_data[x_column].mean()
            mean_y = analysis_data[y_column].mean()
            std_x = analysis_data[x_column].std()
            std_y = analysis_data[y_column].std()

            stats_text = f'X Mean: {mean_x:.2f}, X StdDev: {std_x:.2f}\nY Mean: {mean_y:.2f}, Y StdDev: {std_y:.2f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

            st.pyplot(fig)

            #Correlation Analysis and then I will calculate them
            st.subheader('Correlation Analysis')

            pearson_corr = analysis_data.corr(method='pearson').iloc[0, 1]
            spearman_corr = analysis_data.corr(method='spearman').iloc[0, 1]

            col1, col2 = st.columns(2)
            with col1:
                st.metric('Pearson Correlation', f'{pearson_corr:.4f}')
            with col2:
                st.metric('Spearman Correlation', f'{spearman_corr:.4f}')

            #Model Comparision
            st.subheader('Model Comparision')

            #Function to create pipeline for polynomial regression
            def make_pipeline(poly, regressor):
                class Pipeline:
                    def fit(self, X, y):
                        self.X_poly = poly.fit_transform(X)
                        regressor.fit(self.X_poly, y)
                        self.regressor = regressor
                        self.poly = poly
                        return self

                    def predict(self, X):
                        X_poly = self.poly.transform(X)
                        return self.regressor.predict(X_poly)

                    def score(self, X, y):
                        X_poly = self.poly.transform(X)
                        return self.regressor.score(X_poly, y)

                return Pipeline()

            models = {
                'Linear Regression': LinearRegression(),
                'Polynomial Regression (Degree = 2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                'Polynomial Regression (Degree = 3)': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42)
            }

            #It's time to train my models, and then calculate R^2
            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)  # Train model
                y_pred = model.predict(X_test)  # I'm making predictions
                r2 = r2_score(y_test, y_pred)
                results[name] = r2

            #Display results
            results_df = pd.DataFrame(list(results.items()), columns=['Model', 'R-squared'])
            results_df = results_df.sort_values('R-squared', ascending=False).reset_index(drop=True)

            st.table(results_df)

            #Plot best model
            best_model_name = results_df.iloc[0, 0]
            best_model = models[best_model_name]

            #Create prediction for plotting
            X_plot = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)

            #Get predictions
            if 'Polynomial' in best_model_name:
                degree = int(best_model_name[-2])
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X_scaled)
                lin_reg = LinearRegression()
                lin_reg.fit(X_poly, y_scaled)
                y_pred_plot = lin_reg.predict(poly.transform(X_plot))
            else:
                y_pred_plot = best_model.predict(X_plot)

            #Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_scaled, y_scaled, alpha=0.5, label='Data points')
            ax.plot(X_plot, y_pred_plot, color='red', linewidth=2, label=f'Best model: {best_model_name}')

            if normalize_data:
                ax.set_xlabel(f'{x_column} (Normalized)')
                ax.set_ylabel(f'{y_column} (Normalized)')
            else:
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)

            ax.set_title(f'Best Model Fit: {best_model_name} (R^2 = {results[best_model_name]:.4f})')
            ax.legend()

            st.pyplot(fig)

            #Qualitative Analysis
            st.subheader('Qualitative Analysis')

            #Correlation strength
            if abs(pearson_corr) < 0.3:
                correlation_strength = 'weak'
            elif abs(pearson_corr) < 0.7:
                correlation_strength = 'moderate'
            else:
                correlation_strength = 'strong'

            #Relationship type
            best_linear = best_model_name == 'Linear Regression'
            linear_r2 = results['Linear Regression']
            nonlinear_r2 = max([r2 for name, r2 in results.items() if name != 'Linear Regression'])

            if best_linear or (nonlinear_r2 - linear_r2 < 0.1):
                relationship_type = 'linear'
            else:
                relationship_type = 'non-linear'

            st.write(f'The analysis shows a **{correlation_strength} {relationship_type}** relationship between {x_column} and {y_column}')
            st.write(f'The best model is **{best_model_name}** with R^2 = {results[best_model_name]:.4f}')

            #AI-powered analysis section
            st.subheader('AI-powered Analysis')

            if st.button('Generate AI Analysis'):
                try:
                    analysis_info = {
                        'variable_x': x_column,
                        'variable_y': y_column,
                        'pearson_correlation': pearson_corr,
                        'spearman_correlation': spearman_corr,
                        'best_model': best_model_name,
                        'r_squared': results[best_model_name],
                        'relationship_type': relationship_type,
                        'correlation_strength': correlation_strength,
                    }
                    ai_prompt = f"Variable X: {x_column}, Variable Y: {y_column}, Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}, Best Model: {best_model_name}, R^2: {results[best_model_name]:.4f}, Relationship Type: {relationship_type}, Correlation Strength: {correlation_strength}"
                    st.info(get_ai_analysis(ai_prompt))
                except Exception as e:
                    st.error(f'Failed to generate AI analysis: {e}')

    except Exception as e:
        st.error(f'Failed to load the file: {e}')

#Footer is being added
st.markdown('---')
st.markdown('Built by Madeleine Man Kien Ong with Streamlit, Scikit-learn, and Seaborn (2025)')