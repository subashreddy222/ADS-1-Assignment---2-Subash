import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv_into_dataframe(file_path):
    """
    Reads a CSV file into a DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv('WDIData_T.csv')
    return df

file_path = 'WDIData_T.csv'
df = pd.read_csv(file_path)
# data is stored in a DataFrame called 'df'
df.head()


df.describe()

# Drop rows with missing values
df = df.dropna()

# List of countries to filter
countries_to_filter = ['North Macedonia', 'Norway', 'Oman', 'New Caledonia', 'Nicaragua', 'Nigeria']
indicators_to_filter = ['Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Age dependency ratio (% of working-age population)', 'Agricultural land (% of land area)', 'Agricultural land (sq. km)', 'Cereal yield (kg per hectare)']

# Filter the dataset for the specified countries and indicator Adolescent fertility rate (births per 1,000 women ages 15-19),Age dependency ratio (% of working-age population),Agricultural land (% of land area),Agricultural land (sq. km),Cereal yield (kg per hectare)
filtered_df = df[(df['CountryName'].isin(countries_to_filter)) & (df['IndicatorName'].isin(indicators_to_filter))]

filtered_df

# Line Plot: Age dependency ratio(% of working age population) over the years for different countries
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Value', hue='CountryName', data=filtered_df[filtered_df['IndicatorName'] == 'Age dependency ratio (% of working-age population)'])
plt.title('Age dependency ratio (% of working-age population')
plt.xlabel('Year')
plt.ylabel('Age dependency ratio (% of working-age population')
plt.legend(title='Country')
plt.show()


# Create a pivot table with years as columns and countries as rows
df_years = filtered_df.pivot_table(index=['CountryName', 'CountryCode', 'IndicatorName', 'IndicatorCode'], columns='Year', values='Value').reset_index()

# df_years is the dataframe with years as columns

# Line Plot: Adolescent fertility rate over the years for different countries
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Value', hue='CountryName', data=filtered_df[filtered_df['IndicatorName'] == 'Adolescent fertility rate (births per 1,000 women ages 15-19)'])
plt.title('Adolescent fertility rate (births per 1,000 women ages 15-19)')
plt.xlabel('Year')
plt.ylabel('Adolescent fertility rate')
plt.legend(title='Country')
plt.show()


data = {
    'CountryName': ['Oman', 'Oman', 'Oman', 'Oman', 'Oman'],
    'CountryCode': ['OMN', 'OMN', 'OMN', 'OMN', 'OMN'],
    'IndicatorName': ['Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Age dependency ratio (% of working-age population)', 'Agricultural land (% of land area)', 'Broad money (% of GDP)', 'Arable land (% of land area)'],
    'IndicatorCode': ['SP.ADO.TFRT', 'SP.POP.DPND', 'AG.LND.AGRI.ZS', 'FM.LBL.BMNY.GD.ZS', 'AG.LND.ARBL.ZS'],
    'Year': [1961, 1961, 1961, 1961, 1961],
    'Value': [134.903, 88.84468678, 3.344103393, 5.41182559803319, 0.064620355]
}

df = pd.DataFrame(data)

# Pivot the entire DataFrame, not just for a single year
pivot_df = df.pivot_table(index='CountryName', columns='IndicatorName', values='Value', aggfunc='mean')

# Calculate the correlation matrix
correlation_matrix = pivot_df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)

# Add title and adjust the position
plt.title('Correlation Heatmap for Indicators')

# Show the plot
plt.show()


data = {
    'CountryName': ['North Macedonia', 'Norway', 'Oman', 'New Caledonia', 'Nicaragua', 'Nigeria'],
    'CountryCode': ['MKD', 'NOR', 'OMN', 'NCL', 'NIC', 'NGA'],
    'CO2 Emissions': [4500, 3000, 6000, 8000, 5500, 5000],
    'Arable Land': [30, 20, 10, 15, 25, 12],
    'Forest Area': [20, 15, 30, 10, 18, 22],
    'Rural Population': [5000000, 3000000, 2000000, 4000000, 1000000, 1500000]
}

df_countries = pd.DataFrame(data)

# Set 'CountryName' as the index for correlation calculation
df_countries.set_index('CountryName', inplace=True)

# Select only numeric columns for correlation calculation
numeric_columns = df_countries.select_dtypes(include='number').columns
correlation_matrix = df_countries[numeric_columns].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', linewidths=.5)
plt.title('Correlation Heatmap for Countries Based on Indicators')
plt.show()


df.to_csv('final.csv')


# Sample data
data = {
    'CountryName': ['North Macedonia', 'Norway', 'Oman', 'New Caledonia', 'Nicaragua', 'Nigeria'],
    'CO2 Emissions': [4500, 3000, 6000, 8000, 5500, 5000],
    'Arable Land': [30, 20, 10, 15, 25, 12],
    'Forest Area': [20, 15, 30, 10, 18, 22],
    'Rural Population': [5000000, 3000000, 2000000, 4000000, 1000000, 1500000],
    'Year': [2000, 2001, 2002, 2003, 2004, 2005]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the 'Country Name' column as the index
df.set_index('CountryName', inplace=True)

# Transpose the dataframe for better visualization
df_transposed = df.transpose()

# Drop the row with indicator names

# Plotting different visualizations
plt.figure(figsize=(15, 10))

# Box plot
plt.subplot(2, 2, 3)
sns.boxplot(data=df_transposed.astype(float))
plt.title('Distribution')
plt.xlabel('Country Name')
plt.ylabel('Value')

# Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(df_transposed.astype(float), cmap='YlGnBu', annot=True, fmt=".2f")
plt.title('Multiple Indicators')

plt.tight_layout()
plt.show()


# Provided data
data = {
    'Country Name': ['Africa Eastern and Southern', 'Africa Western and Central', 'Albania'],
    '1972': [515.185447, 80.425799, 568.403202],
    '1973': [538.778544, 87.853116, 593.446746],
    '1974': [564.563028, 89.036474, 591.032643],
    # ... Include data for other years
    '2014': [680.149943, 183.440206, 2309.366503]
}

# Create a DataFrame
df_countries = pd.DataFrame(data)

# Set the 'Country Name' column as the index
df_countries.set_index('Country Name', inplace=True)

# Transpose the dataframe for better visualization
df_countries_transposed = df_countries.transpose()

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_countries_transposed, markers=True)
plt.title('Time Series: Electric Power Consumption (kWh per capita)')
plt.xlabel('Year')
plt.ylabel('Electric Power Consumption (kWh per capita)')
plt.legend(title='Country Name', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


data = {
    'CountryName': ['North Macedonia', 'Norway', 'Oman', 'New Caledonia', 'Nicaragua', 'Nigeria'],
    'CountryCode': ['MKD', 'NOR', 'OMN', 'NCL', 'NIC', 'NGA'],
    'CO2 Emissions': [4500, 3000, 6000, 8000, 5500, 5000],
    'Arable Land': [30, 20, 10, 15, 25, 12],
    'Forest Area': [20, 15, 30, 10, 18, 22],
    'Rural Population': [5000000, 3000000, 2000000, 4000000, 1000000, 1500000],
    'Year': [2000, 2001, 2002, 2003, 2004, 2005]
}

df_countries = pd.DataFrame(data)
# Set the 'Country Name' column as the index
df_countries.set_index('CountryName', inplace=True)

# Transpose the dataframe for better visualization
df_countries_transposed = df_countries.transpose()

# Plot time series for each indicator
plt.figure(figsize=(12, 8))
sns.lineplot(data=df_countries, markers=True, dashes=False)
plt.title('Time Series Plot for Selected Countries and Indicators')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
