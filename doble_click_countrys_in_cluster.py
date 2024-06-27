#%%
# Library
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#%%

# Define the list of countries and years (The selected from clustering)
countries = ['Austria', 'Belgium', 'Denmark', 'Netherlands', 'Sweden']
years = list(range(2005, 2016))

# Path to the directory containing the Excel files
directory_path = "C:/Users/Aldis/Documents/Master Data Science/GitHub/Urban_Metabolism_24/Data/Week_2_Data/PIOT_country"

# Initialize an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()
#%%

# Iterate through the years and countries to construct the filenames
for year in years:
    for country in countries:
        filename = f"{year}_{country}_PIOT_balanced[tonnes_cap]_simpnumv13.xlsx"
        file_path = os.path.join(directory_path, filename)

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the data from the "PIOT" sheet
            try:
                df = pd.read_excel(file_path, sheet_name='PIOT')
                df['Year'] = year
                df['Country'] = country
                # Filter rows where the first column is "DE" or "IMP"
                filtered_df = df[df.iloc[:, 0].isin(['DE', 'IMP'])]
                # Append the filtered data to the combined DataFrame
                combined_df = pd.concat([combined_df, filtered_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

# Check for duplicates
combined_df.drop_duplicates(inplace=True)

# Ensure only the required columns are selected
columns = ['Country', 'Year', 'AGRICUL', 'MININGFF', 'MININGMM', 'MININGNM', 'FOOD', 'TEXTILES', 'WOOD',
           'PAPER', 'PETROLPROD', 'CHEM&PHARM', 'PLASTICS', 'OTHNM', 'METALSBASIC', 'METALPROD', 'ELECTRONICS',
           'ELEC.EQUIP', 'MACHINERY', 'VEHICLES', 'TRANSPOTH', 'MANUFOTH', 'UTILITIES', 'CONSTRUC', 'SERVICES']

final_df = combined_df[[col for col in columns if col in combined_df.columns]]

# Ensure no duplicated columns
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
print(final_df)
#%%

# Group by country and year, and sum the columns
summed_df = final_df.groupby(['Country', 'Year']).sum().reset_index()
print(summed_df)
# %%

# Combine MININGFF, MININGMM, and MININGNM into a single column MINING
summed_df['MINING'] = summed_df[['MININGFF', 'MININGMM', 'MININGNM']].sum(axis=1)
# Drop the original MININGFF, MININGMM, and MININGNM columns
summed_df.drop(columns=['MININGFF', 'MININGMM', 'MININGNM'], inplace=True)

print(summed_df)
#%%

# Open SEM_Variables.xlsx
sem_file_location = "C:/Users/Aldis/Documents/Master Data Science/GitHub/Urban_Metabolism_24/Data/Week_2_Data/SEM_data/SEM_Variables.xlsx"
sem_data = pd.read_excel(sem_file_location)

# Create a data frame od sem_data
sem_df_2 = pd.DataFrame(sem_data)
sem_df_2.head()
#%%

# Define the list of countries and years (The selected from clustering)
countries = ['Austria', 'Belgium', 'Denmark', 'Netherlands', 'Sweden']
years = list(range(2005, 2016))

# Define columns we want to keep
columns_to_keep = [
    'Year', 'Country', 'GDP in 2015 EUR/inhab', 'RP_AGRICUL', 'RP_MINING', 'RP_FOOD', 'RP_TEXTILES',
    'RP_WOOD', 'RP_PAPER', 'RP_PETROLPROD', 'RP_CHEM&PHARM', 'RP_PLASTICS', 'RP_OTHNM',
    'RP_METALSBASIC', 'RP_METALPROD', 'RP_ELECTRONICS', 'RP_ELEC.EQUIP', 'RP_MACHINERY', 'RP_VEHICLES',
    'RP_TRANSPOTH', 'RP_MANUFOTH', 'RP_UTILITIES', 'RP_CONSTRUC', 'RP_SERVICES'
]

# Filter
filtered_df = sem_df_2[(sem_df_2['Year'].isin(years)) & (sem_df_2['Country'].isin(countries))]
filtered_df = filtered_df[columns_to_keep]
print(filtered_df)
#%%

# Merge the 2 DataFrames with `Country` and `Year`
merged_df = pd.merge(summed_df, filtered_df, on=['Country', 'Year'])

# List of the columns that need to be multiplied
columns_to_multiply = [
    'AGRICUL', 'FOOD', 'TEXTILES', 'WOOD', 'PAPER', 'PETROLPROD',
    'CHEM&PHARM', 'PLASTICS', 'OTHNM', 'METALSBASIC', 'METALPROD',
    'ELECTRONICS', 'ELEC.EQUIP', 'MACHINERY', 'VEHICLES', 'TRANSPOTH',
    'MANUFOTH', 'UTILITIES', 'CONSTRUC', 'SERVICES', 'MINING'
]

# New DataFrame with the columns `Country`, `Year` and `GDP in 2015 EUR/inhab`
result_df = merged_df[['Country', 'Year', 'GDP in 2015 EUR/inhab']].copy()

# Multiplication
for col in columns_to_multiply:
    result_df[col] = merged_df[col] * merged_df['RP_' + col]
    
result_df['Total_Value'] = result_df.iloc[:, 3:].sum(axis=1)

print(result_df)
#%%

# Porcentaje of economic sector
calculated_df = pd.DataFrame({
    'Country': result_df['Country'],
    'Year': result_df['Year']
})

for col in columns_to_multiply:
    calculated_df[col] = (result_df[col] / result_df['Total_Value']) * result_df['GDP in 2015 EUR/inhab']

print(calculated_df)
#%%

# Plot of principal economic sectors in each contry 
# Contry list
countries = calculated_df['Country'].unique()

# Plot configuration
fig, axes = plt.subplots(nrows=len(countries), figsize=(15, 8 * len(countries)), sharex=True)

# Each year with different color
colors = plt.get_cmap('tab10', len(summed_df['Year'].unique()))

# For each country
for i, country in enumerate(countries):
    # Filter
    country_data = calculated_df[calculated_df['Country'] == country]

    years = country_data['Year']
    sectors = country_data.columns[2:]  

    # Bar plot
    bar_width = 0.2 
    index = np.arange(len(sectors))  

    for j, year in enumerate(years):
        # Position of bar for each year
        bars = axes[i].bar(index + j * bar_width, country_data.iloc[j, 2:], bar_width,
                           label=year, color=colors(j))

    # Plot configuration
    axes[i].set_title(f"Economic Sector Distribution - {country}")
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_xticks(index + (len(years) - 1) * bar_width / 2)
    axes[i].set_xticklabels(sectors, rotation=45)
    axes[i].legend()
    axes[i].grid(True)

    axes[i].tick_params(axis='x', rotation=90)

# Plot
plt.tight_layout()
plt.show()
#%%



# Random Forest Algorithm
# Country list
countries = calculated_df['Country'].unique()

# Initialize an empty DataFrame to store top 5 features for each country
top_features_df = pd.DataFrame()

# Iterate over each country and train a separate Random Forest model
for country in countries:
    # Filter country data
    country_data = calculated_df[calculated_df['Country'] == country]
    X = country_data.drop(['Country', 'Year'], axis=1)
    y = result_df[result_df['Country'] == country]['Total_Value']
    
    # Colors for the economic sectors
    palette = sns.color_palette("tab10", len(X.columns))
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{country} - MSE: {mse}, R²: {r2}")
    
    # Extract the importance of features
    importances = model.feature_importances_
    features = X.columns
    
    # Create a DataFrame with feature importances
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    # Plot features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette=palette[:len(feature_importances)])
    plt.title(f'Importancia de las Características - {country}')
    plt.show()
    
    # Get the top 5 most important features for current country
    top_5_features = feature_importances.head(5)
    
    # Add country column to top_5_features DataFrame
    top_5_features['Country'] = country
    
    # Append to top_features_df
    top_features_df = pd.concat([top_features_df, top_5_features], ignore_index=True)

# Calculate frequency of each feature across countries
top_feature_freq = top_features_df['Feature'].value_counts().reset_index()
top_feature_freq.columns = ['Feature', 'Frequency']

# Select top 5 features with highest frequency
top_5_consistent_features = top_feature_freq.head(5)

# Print top 5 consistent features
print("Top 5 Consistent Features across Countries:")
print(top_5_consistent_features)
print("\n")

# Plot top 5 consistent features
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Frequency', data=top_5_consistent_features, palette='viridis')
plt.title('Top 5 Consistent Features across Countries')
plt.xlabel('Feature')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%


# OLR Regresson
# Top 5 consistent features
features = top_5_consistent_features['Feature'].values
X = calculated_df[features]

# Add a constant (intercept)
X = sm.add_constant(X)
# GDP of each country per year
y = result_df['Total_Value']

# Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OLS model
model = sm.OLS(y_train, X_train).fit()
# Resume
print(model.summary())

# Evaluate in train data
y_pred = model.predict(X_test)

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# %%




