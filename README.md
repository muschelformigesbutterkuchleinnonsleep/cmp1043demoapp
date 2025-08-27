# cmp1043demoapp

Hey, welcome! I'm Madeleine Man Kien Ong, and it's so nice to see you around ehe.

You should install these required packages: 
`streamlit`
`pandas`
`numpy`
`matplotlib`
`seaborn`
`scikit-learn`

## Features of the Streamlit Web App
This app provides a comprehensive data analysis tool with a minimalist interface. Here are its key features:

### 1. Data Upload & Selection
Users can upload any CSV file.
The app displays a preview of the data and summary statistics.
Users can select any two numeric variables for analysis.

### 2. Data Processing Options
**Missing value handling**: Fill with mean or drop rows.
**Outlier removal**: Using the IQR method.
**Data normalization/scaling**: Optional.

### 3. Data Visualization
Interactive scatter plots showing the relationship between variables.
Statistical information displayed on the plot (mean, standard deviation).

### 4. Statistical Analysis
Calculates both Pearson and Spearman correlation coefficients.
Displays the correlation strength in an easy-to-understand format.

### 5. Model Comparison
Compares multiple regression models: Linear Regression, Polynomial Regression (degree 2 and 3), Decision Tree Regression, Random Forest Regression
Automatically ranks models by R-squared value.
Plots the best model's prediction line over the data.

### 6. Automated Insights
Automatically determines if the relationship is weak, moderate, or strong.
Identifies if the relationship is linear or non-linear.
Generates natural language summaries of the findings.

### 7. AI-powered Analysis
Includes a placeholder for API integration with ChatGPT/Claude/DeepSeek.
Shows how the detailed analysis would appear when generated.
