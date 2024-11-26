# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pulp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os 

def load_dataset():
    # Load your dataset
    df = pd.read_csv('./datasets/daily_dataset.csv')

    # Convert 'day' column to datetime
    df['day'] = pd.to_datetime(df['day'], format='%d-%m-%Y')


    # Extract features
    df['day_of_week'] = df['day'].dt.dayofweek
    df['month'] = df['day'].dt.month
    df['day_of_month'] = df['day'].dt.day
    df['year'] = df['day'].dt.year

    # Convert LCLid to categorical codes
    df['LCLid'] = df['LCLid'].astype('category').cat.codes

    df.dropna(inplace=True)
    
    return df

def train_model(df):
    # Define features and target
    X = df[['LCLid', 'day_of_week', 'month', 'day_of_month', 'year']]
    y = df['energy_median']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.2)
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_xgb)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error: {rmse}')
    
    r2 = r2_score(y_test, y_pred_xgb)
    print(f'R-squared: {r2}')
    
    return xgb_model

# Calculate predicted demand for each area on the same day in previous years
def calculate_predicted_demand(df, selected_date, model):
    # Ensure selected_date is in datetime format
    selected_date = pd.to_datetime(selected_date)
    
    predicted_demand = []
    for lclid in df['LCLid'].unique():
        # Extract features for the selected date
        features = {
            'LCLid': [lclid],  # Include LCLid as a feature
            'day_of_week': [selected_date.dayofweek],
            'month': [selected_date.month],
            'day_of_month': [selected_date.day],
            'year': [selected_date.year]
        }
        features_df = pd.DataFrame(features)
        
        # Predict demand for the locality using the trained model
        prediction = model.predict(features_df)[0]
        predicted_demand.append({'LCLid': lclid, 'predicted_energy': prediction})
    
    # Convert the list of dictionaries to a DataFrame
    predicted_demand_df = pd.DataFrame(predicted_demand)
    return predicted_demand_df

# Function to solve the LP problem with weighted allocation
import pulp

import pulp

def solve_lp_problem(area_avg_demand, total_generated):
    # Initialize the LP problem
    print(total_generated)
    print(type(total_generated))
    total_generated=int(total_generated)
    lp_problem = pulp.LpProblem("Electricity_Distribution", pulp.LpMinimize)

    # Define decision variables for each area
    allocation = {area: pulp.LpVariable(f"allocation_{area}", lowBound=0) for area in area_avg_demand.index}

    # Objective function: Minimize the total amount of allocated energy (just to define a solvable LP)
    lp_problem += pulp.lpSum(allocation.values()), "Total Allocation"

    # Add constraint: Total allocated energy should not exceed the total generated energy
    lp_problem += pulp.lpSum(allocation.values()) <= total_generated, "Energy Constraint"

    # Add constraints: Allocate energy proportionally based on historical average demand
    total_avg_demand = area_avg_demand.sum()
    for area, avg_demand in area_avg_demand.items():
        # Ensure each area gets at least a proportionate share of energy
        proportionate_share = (avg_demand / total_avg_demand) * total_generated
        lp_problem += allocation[area] >= proportionate_share, f"MinAllocation_{area}"

    # Solve the problem
    lp_problem.solve()

    # Return the allocation results
    result = {area: allocation[area].varValue for area in area_avg_demand.index}
    return result



# Compare the allocated energy with historical demand
def evaluate_allocation(allocation_result, area_avg_demand_same_day):
    sufficiency_report = {}
    for area, allocated_energy in allocation_result.items():
        historical_demand = area_avg_demand_same_day.get(area, 0)
        sufficiency = "Sufficient" if allocated_energy >= historical_demand else "Insufficient"
        sufficiency_report[area] = {
            "Allocated Energy": allocated_energy,
            "Historical Demand": historical_demand,
            "Sufficiency": sufficiency
        }
    return sufficiency_report

def distribute_energy(total_generated, predicted_demand_df):
    
    # Convert the predicted demand DataFrame into a Series for compatibility
    area_avg_demand_same_day = predicted_demand_df.set_index('LCLid')['predicted_energy']

    allocation_result = solve_lp_problem(area_avg_demand_same_day, total_generated)
    sufficiency_report = evaluate_allocation(allocation_result, area_avg_demand_same_day)

    allocation_result = pd.DataFrame(list(allocation_result.items()), columns=['LCLid', 'allocated_energy'])
    return allocation_result

def save_map_image(gdf, column_name, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax, column=column_name, cmap='Reds', legend=True, missing_kwds={"color": "lightgrey"})
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def generate_prediction_map(predicted_demand_df):
    shapefile_path = './datasets/london_shapefile/london.shp'
    gdf = gpd.read_file(shapefile_path)

    # Load the datasets
    daily_dataset = predicted_demand_df
    synthetic_coordinates = pd.read_csv('./datasets/synthetic_locality_coordinates.csv')

    # Randomly select 983 LCLid entries
    selected_lclids = daily_dataset['LCLid'].sample(n=983, random_state=40)

    # Merge with the energy data to get the relevant columns
    selected_data = daily_dataset[daily_dataset['LCLid'].isin(selected_lclids)]

    # Merge synthetic coordinates with selected data
    selected_coordinates = synthetic_coordinates[synthetic_coordinates['LCLid'].isin(selected_lclids)]
    selected_gdf = gpd.GeoDataFrame(selected_coordinates, geometry=gpd.points_from_xy(selected_coordinates.longitude, selected_coordinates.latitude))

    # Assign the selected points to the polygons
    gdf['energy_mean'] = np.nan  # Initialize with NaN

    for i in range(len(gdf)):
        gdf.at[i, 'energy_mean'] = selected_data.iloc[i]['predicted_energy']

    # Save the plot as an image
    filename = './static/prediction_map.png'
    save_map_image(gdf, 'energy_mean', 'Heatmap of Predicted Energy Usage in London', filename)
    return filename

def generate_distribution_map(allocation_result_df):
    shapefile_path = './datasets/london_shapefile/london.shp'
    gdf = gpd.read_file(shapefile_path)

    # Load the datasets
    daily_dataset = allocation_result_df
    synthetic_coordinates = pd.read_csv('./datasets/synthetic_locality_coordinates.csv')

    # Randomly select 983 LCLid entries
    selected_lclids = daily_dataset['LCLid'].sample(n=983, random_state=40)

    # Merge with the energy data to get the relevant columns
    selected_data = daily_dataset[daily_dataset['LCLid'].isin(selected_lclids)]

    # Merge synthetic coordinates with selected data
    selected_coordinates = synthetic_coordinates[synthetic_coordinates['LCLid'].isin(selected_lclids)]
    selected_gdf = gpd.GeoDataFrame(selected_coordinates, geometry=gpd.points_from_xy(selected_coordinates.longitude, selected_coordinates.latitude))

    # Assign the selected points to the polygons
    gdf['energy_mean'] = np.nan  # Initialize with NaN

    for i in range(len(gdf)):
        gdf.at[i, 'energy_mean'] = selected_data.iloc[i]['allocated_energy']

    # Save the plot as an image
    filename = './static/distribution_map.png'
    save_map_image(gdf, 'energy_mean', 'Heatmap of Energy Distribution in London', filename)
    return filename
