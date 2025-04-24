import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.eval_measures import aic, bic
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import os
#import streamlit as st

def detect_date_column(df: pd.DataFrame) -> str:
    """
    Automatically detects the column in the DataFrame that contains date-like information,
    including quarterly data in the format "YYYY Qx".

    Parameters:
        df (pd.DataFrame): The DataFrame to search for date columns.

    Returns:
        str: The name of the detected date column, or None if no date column is found.
    """
    for column in df.columns:
        try:
            # Convert column to string if it's not already a string type
            if df[column].dtype != 'O':  # Not object type (string)
                df[column] = df[column].astype(str)

            # Check if column values are in "YYYY Qx" format (e.g., "2021 Q1")
            if df[column].str.contains(r'\d{4} Q[1-4]', na=False).any():
                return column  # Return the column name if it contains "YYYY Qx" format

            # Try to convert the column to datetime for standard date formats
            pd.to_datetime(df[column], errors='raise')
            return column  # Return the column name if it's a valid date column
        except (ValueError, TypeError):
            continue
    return None  # Return None if no valid date column is found

def convert_quarter_to_date(quarter_str: str) -> pd.Timestamp:
    """
    Converts a quarterly string (e.g., "2021 Q1") to the corresponding date (e.g., "2021-01-01").
    This function is flexible and can work with any quarterly string containing "YYYY Qx".

    Parameters:
        quarter_str (str): The quarterly string (e.g., "2021 Q1").

    Returns:
        pd.Timestamp: The corresponding datetime object (e.g., "2021-01-01").
    """
    # Split the string into year and quarter parts
    parts = quarter_str.split(" Q")
    if len(parts) == 2:
        year, quarter = parts
        # Handle the case where the string contains a valid year and quarter
        try:
            year = int(year)
            quarter = int(quarter)
            # Map quarters to the first month of each quarter
            quarter_months = {1: 1, 2: 4, 3: 7, 4: 10}
            if quarter in quarter_months:
                month = quarter_months[quarter]
                return pd.to_datetime(f"{year}-{month:02d}-01")
        except ValueError:
            pass  # If parsing fails, return NaT (could be invalid data)
    return pd.NaT  # Return NaT if the input string is not in a valid "YYYY Qx" format

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by stripping column names and converting date columns to datetime format.
    If a column is detected as a date, it is renamed to 'Date'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    date_column = detect_date_column(df)

    if date_column:
        # Check if the column contains quarterly-like data (e.g., "YYYY Qx")
        if df[date_column].dtype == object and df[date_column].str.contains(r'\d{4} Q[1-4]', na=False).any():
            # Convert the "YYYY Qx" format to datetime dynamically
            df[date_column] = df[date_column].apply(convert_quarter_to_date)
        else:
            # For standard date formats, convert to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Rename the date column to 'Date'
        df.rename(columns={date_column: 'Date'}, inplace=True)

    return df

def load_dataset(folder_path: str = "data", crisis_years: list = []) -> dict:
    """
    Loads all CSV datasets from a specified folder, applies cleaning operations,
    and splits them based on specified crisis years.

    Parameters:
        folder_path (str): The path to the folder containing CSV files. Defaults to "data".
        crisis_years (list): List of years representing the crisis periods (e.g., [2008, 2022]).

    Returns:
        dict: A dictionary where the keys are crisis years (e.g., 2008, 2022) and the values
              are lists of DataFrames corresponding to each crisis year.
    """
    crisis_data = {year: [] for year in crisis_years}  # Initialize a list for each crisis year

    # Iterate over all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Clean the dataset
            df = clean_dataset(df)

            # Automatically detect the date column
            date_column = detect_date_column(df)

            if date_column:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')  # Convert to datetime
                df["Year"] = df[date_column].dt.year

                # Check if the dataset contains any data from the crisis years
                unique_years = df["Year"].unique()  # Get unique years in dataset
                for year in crisis_years:
                    if year in unique_years:
                        crisis_data[year].append(df)  # Add entire dataset to corresponding crisis year

                df.drop(columns=["Year"], inplace=True)  # Remove temporary Year column

    return crisis_data

def normalisation(df, normalization_type='z-score', columns=None):
    """
    Normalizes the numeric columns of a DataFrame using the specified normalization method.

    Args:
    - df (pd.DataFrame): The DataFrame to normalize.
    - normalization_type (str): The type of normalization to apply. Options are 'z-score' or 'min-max'. Default is 'z-score'.
    - columns (list or None): List of column names to normalize. If None, normalizes all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with normalized values.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if columns is not None:
        # Use provided columns, ensuring they are numeric
        numeric_columns = [col for col in columns if col in numeric_columns]

    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns to normalize in the provided DataFrame.")

    # Select the normalization method
    if normalization_type == 'z-score':
        scaler = StandardScaler()
    elif normalization_type == 'min-max':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}. Supported types are 'z-score' and 'min-max'.")

    # Apply the scaler to the numeric columns
    normalized_data = scaler.fit_transform(df[numeric_columns])

    # Convert the normalized data back to a DataFrame
    normalized_df = df.copy()
    normalized_df[numeric_columns] = normalized_data

    return normalized_df

def adjust_column_for_periods(df: pd.DataFrame, periods: list):
    """
    Adjusts specified columns in a DataFrame based on multiple periods with varying start/end dates and values.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - periods (list): A list of dictionaries, where each dictionary contains:
        - 'start': start date of the period (str or datetime)
        - 'end': end date of the period (str or datetime)
        - 'column': the name of the column to adjust (str)
        - 'value': the adjustment value to apply during the period (int, float, or str)

    Returns:
    - pd.DataFrame: The adjusted DataFrame.
    """
    # Copy the original DataFrame to avoid modifying the input DataFrame directly
    adjusted_df = df.copy()

    for period in periods:
        # Convert start and end dates to datetime
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])

        # Check if the column exists in the DataFrame
        column = period['column']
        if column not in adjusted_df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        # Apply the adjustment based on the period
        adjusted_df.loc[
            (adjusted_df['Date'] >= start_date) &
            (adjusted_df['Date'] <= end_date),
            column
        ] = period['value']

    return adjusted_df

def preprocess_data(df: pd.DataFrame, freq: str = 'MS', method: str = 'ffill') -> pd.DataFrame:
    """
    Preprocesses a DataFrame by converting date formats and resampling it to a given frequency.

    This version supports various types of date formats and resampling methods.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        freq (str): The resampling frequency. Default is 'MS' (monthly start).
                    Other options include 'D' (daily), 'M' (monthly), 'Y' (yearly), etc.
        method (str): Method for filling missing data during resampling. Default is 'ffill' (forward fill).
                      Other options: 'bfill' (backward fill), 'pad' (pad), 'nearest', etc.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with resampled data.
    """
    df = df.dropna(subset=['Date'])

    df_resampled = df.set_index('Date').resample(freq).apply(method).reset_index()

    return df_resampled

def merge_dataframes_on_date(*dfs):
    """
    Merges multiple dataframes on the 'Date' column.

    Parameters:
        *dfs: Multiple DataFrames to merge.

    Returns:
        pd.DataFrame: A single merged DataFrame on the 'Date' column.
    """
    # Preprocess each DataFrame before merging
    processed_dfs = [preprocess_data(df.copy()) for df in dfs]

    # Merge DataFrames on 'Date'
    merged_df = processed_dfs[0]
    for df in processed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Date', how='inner')

    return merged_df

def filter_target_column(all_dfs, crisis_year, target_column):
    """
    Filters each DataFrame in the all_dfs list for a given 'target_column' and 'Date' column.

    Parameters:
        all_dfs (list): List of DataFrames to filter.
        crisis_year (str): The year or key indicating which group of DataFrames to filter.
        target_column (str): The column to filter along with 'Date'.

    Returns:
        None: Modifies the DataFrames in place.
    """
    # Loop through each DataFrame in the selected crisis_year
    for i, df in enumerate(all_dfs[crisis_year]):
        if target_column in df.columns:
            # Filter the DataFrame to only include 'Date' and the target column
            df_filtered = df[['Date', target_column]].copy()

            # Optionally, assign it back to the list to modify in place
            all_dfs[crisis_year][i] = df_filtered

def plotting(df: pd.DataFrame,
             step_column=" ",
             title: str = None,
             x_label: str = None,
             y_label: str = None,
             annotation: dict = None,
             highlight: list = None):
    """
    Plots a time series graph for a dataset with a detected date column.
    Supports step charts, event highlighting, and shaded EPG periods.

    Parameters:
        df (pd.DataFrame): The dataset containing at least one date column and one value column.
        step_column (str): The column to be plotted as a step chart.
        title (str): Title of the plot (default: "Time Series Plot").
        x_label (str, optional): Label for the x-axis. Defaults to the detected date column name.
        y_label (str, optional): Label for the y-axis. Defaults to the value column names.
        annotation (dict, optional): {"Event Name": ("YYYY-MM-DD", with_line)}.
                                           If with_line=True, a vertical line is drawn.
        highlight (list of tuples, optional): List of EPG periods in format:
            [(start_date, end_date, price_level, label, color), ...].
        annotation_arrows (bool, optional): If True, event labels will be placed without vertical lines.
    """
    date_col = 'Date'
    if df.shape[1] < 2:
        raise ValueError("The DataFrame must contain at least two columns: a date column and at least one value column.")

    # Default axis labels
    if x_label is None:
        x_label = date_col
    if y_label is None:
        y_label = ", ".join(df.columns[df.columns != date_col])

    # Default title
    if title is None:
        title = y_label

    fig, ax1 = plt.subplots(figsize=(15, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Plotting main data
    for i, column in enumerate(df.columns[df.columns != date_col]):
        line_style = '-'
        marker_style = 'o'

        if step_column == column:
            ax1.step(df[date_col], df[step_column], where='post', linestyle='-',
                     color=colors[i % len(colors)], linewidth=2, label=f'{step_column} (Step)')
            continue

        ax1.plot(df[date_col], df[column], linestyle=line_style, marker=marker_style,
                 linewidth=2, markersize=6, color=colors[i % len(colors)], label=column)

    # Plot shaded bars
    if highlight:
        for start, end, price, label, color in highlight:
            width = (end - start).days
            middle = start + (end - start) / 2

            ax1.bar(x=start, height=price, width=width, color=color, alpha=0.2, align='edge', label=label)
            ax1.annotate(label, xy=(middle, price), xytext=(middle, price + 200),
                         arrowprops=dict(arrowstyle='->', color=color), color=color, fontsize=12, ha='center')

    # Highlight specific events (both with and without lines)
    if annotation:
        for event_name, (event_date, with_line) in annotation.items():
            event_date = pd.to_datetime(event_date)

            # Draw vertical line if required
            if with_line:
                ax1.axvline(x=event_date, color='red', linestyle='-', linewidth=2, label=event_name)

            # Annotate the event
            y_position = df.iloc[:, 1].max() - (200 if with_line else 100)

            # Annotate the event above the data points
            ax1.annotate(event_name,
             xy=(event_date, y_position),  # Annotation point
             xytext=(event_date + pd.DateOffset(days=150), y_position + 200),  # Offset text
             fontsize=12, color='red',
             ha='center',
             arrowprops=dict(arrowstyle='->', color='blue' if not with_line else 'red'))



    # Labels, title, and legend
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_correlation_matrix(df: pd.DataFrame, title: str):
    """
    Plots a heatmap for the correlation matrix of all numeric columns (excluding 'Date') in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing the 'Date' column and numeric columns.
        title (str): The title of the plot.
    """
    # Exclude the 'Date' column and select numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Calculate the correlation matrix for the numeric columns
    correlation_matrix = df[numeric_columns].corr()

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
    plt.title(title)
    plt.show()

class GrangerCausalityAnalyzer:
    def __init__(self, data, maxlag=2, adf_significance=0.05, causality_significance=0.05):
        """
        Initializes the Granger Causality Analyzer.

        Parameters:
        - data: DataFrame containing the time-series variables.
        - maxlag: Maximum number of lags for Granger Causality tests (default=4).
        - adf_significance: Significance level for ADF test (default=0.05).
        - causality_significance: Significance level for Granger Causality test (default=0.05).
        """
        self.data = data
        self.maxlag = maxlag
        self.adf_significance = adf_significance
        self.causality_significance = causality_significance
        self.stationary_vars = {}
        self.stationarity_summary = {}
        self.causality_results = []

    def adf_test(self, series, name):
        """Performs the Augmented Dickey-Fuller (ADF) test for stationarity."""
        result = adfuller(series.dropna(), autolag='AIC')
        print(f'ADF Test for {name}:')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f'Critical Values: {result[4]}')

        if result[1] <= self.adf_significance:
            print(f'{name} is stationary (reject null hypothesis)\n')
            return True, result[1]
        else:
            print(f'{name} is non-stationary (fail to reject null hypothesis)\n')
            return False, result[1]

    def transform_to_stationary(self):
        """Transforms non-stationary variables to stationary using differencing."""
        variables = [col for col in self.data.columns if col != 'Date']
        data_transformed = self.data[variables].copy()

        for var in variables:
            is_stationary, p_value = self.adf_test(data_transformed[var], var)
            if is_stationary:
                self.stationary_vars[var] = var
                self.stationarity_summary[var] = f"stationary (p-value = {p_value:.4f})"
            else:
                # First difference
                data_transformed[f'{var}_diff'] = data_transformed[var].diff()
                is_stationary, p_value = self.adf_test(data_transformed[f'{var}_diff'], f'Differenced {var}')
                if is_stationary:
                    self.stationary_vars[var] = f'{var}_diff'
                    self.stationarity_summary[var] = f"non-stationary, stationary after first differencing (p-value = {p_value:.4f})"
                else:
                    # Second difference
                    data_transformed[f'{var}_diff2'] = data_transformed[f'{var}_diff'].diff()
                    is_stationary, p_value = self.adf_test(data_transformed[f'{var}_diff2'], f'Second Differenced {var}')
                    if is_stationary:
                        self.stationary_vars[var] = f'{var}_diff2'
                        self.stationarity_summary[var] = f"non-stationary, stationary after second differencing (p-value = {p_value:.4f})"
                    else:
                        self.stationary_vars[var] = f'{var}_diff2'
                        self.stationarity_summary[var] = f"non-stationary even after second differencing (p-value = {p_value:.4f})"

        # Create final stationary dataset
        stationary_columns = [self.stationary_vars[var] for var in variables]
        data_stationary = data_transformed[stationary_columns].dropna()
        data_stationary.columns = variables  # Rename columns to original names for clarity
        self.data_stationary = data_stationary

        print("Stationary dataset created:")
        print(self.data_stationary.head())
        print("\nNaN values in stationary dataset:")
        print(self.data_stationary.isna().sum())

        return self.data_stationary

    def run_granger_causality_tests(self):
        """Runs Granger Causality tests on stationary variables."""
        print("\nRunning Granger Causality Tests...\n")
        variables = list(self.stationary_vars.keys())

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    print(f'Granger Causality Test: {var2} -> {var1}')
                    test_results = grangercausalitytests(self.data_stationary[[var1, var2]], maxlag=self.maxlag, verbose=True)

                    # Parse the results dictionary
                    for lag, result in test_results.items():
                        p_value = result[0]['ssr_ftest'][1]  # Extract p-value
                        if p_value <= self.causality_significance:
                            self.causality_results.append({'cause': var2, 'effect': var1, 'lag': lag, 'p_value': p_value})

                    print("\n" + "="*50 + "\n")

    def generate_conclusion(self):
        """Generates a summary of the findings from the Granger Causality tests."""
        print("\nConclusion: Granger Causality Analysis of Economic Indicators\n")

        # Stationarity Summary
        print("Stationarity Analysis:")
        for var, summary in self.stationarity_summary.items():
            print(f"- {var}: {summary}")
        print("\n")

        # Granger Causality Findings
        print("Granger Causality Findings:")
        if self.causality_results:
            for result in self.causality_results:
                print(f"- {result['cause']} Granger-causes {result['effect']} at lag {result['lag']} (p-value = {result['p_value']:.4f})")
        else:
            print("- No significant Granger Causality relationships were found.")
        print("\n")

    def run_workflow(self):
        """Executes the full workflow: transformation, Granger Causality, and conclusion."""
        self.transform_to_stationary()
        self.run_granger_causality_tests()
        self.generate_conclusion()

def lasso_feature_selection(df, target_column='GDP', test_size=0.2, random_state=42):
    """
    Performs LASSO regression for feature selection on a dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing economic indicators and a target variable.
        target_column (str): The name of the column to predict (e.g., 'GDP').
        test_size (float): The proportion of the dataset to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        list: Selected features from LASSO.
    """
    # Drop target and non-numeric columns
    X = df.drop(columns=[target_column, 'Date'], errors='ignore')
    y = df[target_column]

    # Handle missing values by filling with the median
    X.fillna(X.median(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # Define hyperparameter range for alpha
    params = {"alpha": np.linspace(0.00001, 10, 500)}

    # Use K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Initialize LASSO and perform GridSearchCV
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(X_scaled, y)

    # Get best alpha parameter
    best_alpha = lasso_cv.best_params_["alpha"]
    print(f"Best Alpha: {best_alpha}")

    # Train LASSO with best alpha
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X_train, y_train)

    # Select important features (non-zero coefficients)
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    print("Selected Features by LASSO:", selected_features)

    # Plot feature importance
    plt.figure(figsize=(15, 8))
    plt.bar(selected_features, lasso.coef_[lasso.coef_ != 0], color='blue')
    plt.xlabel("Features")
    plt.ylabel("LASSO Coefficient")
    plt.title("Feature Importance from LASSO Regression")
    plt.xticks(rotation=45)
    plt.show()

    return selected_features

class TimeSeriesRegression:
    def __init__(self, df, features, target, max_lag=6):
        """
        Initializes the TimeSeriesRegression model.

        Parameters:
            df (pd.DataFrame): The dataset.
            features (list): List of feature column names.
            target (str): Target variable.
            max_lag (int): Maximum lag to consider.
        """
        self.df = df.copy()
        self.features = features
        self.target = target
        self.max_lag = max_lag
        self.best_lags = {}

    def find_best_lags(self):
        """Finds the optimal lag for each feature using time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)

        for feature in self.features:
            best_mse = float('inf')
            best_lag = 0

            for lag in range(1, self.max_lag + 1):
                df_lagged = self.df[[feature, self.target]].copy()
                df_lagged[f"{feature}_lag"] = df_lagged[feature].shift(lag)
                df_lagged.dropna(inplace=True)

                X = df_lagged[[f"{feature}_lag"]]
                y = df_lagged[self.target]

                mse_list = []

                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    mse_list.append(mse)

                avg_mse = np.mean(mse_list)

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_lag = lag

            self.best_lags[feature] = best_lag  # Store best lag

    def create_lagged_dataset(self):
        """Creates a DataFrame with the best lagged features."""
        self.df_lagged = self.df.copy()
        for feature, lag in self.best_lags.items():
            self.df_lagged[f"{feature}_lag"] = self.df_lagged[feature].shift(lag)

        self.df_lagged.dropna(inplace=True)  # Remove NaN rows
        return self.df_lagged

    def train_model(self):
        """Trains a multiple linear regression model using the best lags."""
        X = self.df_lagged[[f"{feat}_lag" for feat in self.best_lags.keys()]]
        y = self.df_lagged[self.target]

        self.model = LinearRegression()
        self.model.fit(X, y)

        # Predict & Evaluate
        self.y_pred = self.model.predict(X)
        self.mse = mean_squared_error(y, self.y_pred)
        print(f"Final Model MSE: {self.mse}")

    def plot_actual_vs_predicted(self):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df_lagged[self.target], self.y_pred, color='blue', alpha=0.6, label="Actual vs Predicted")
        plt.plot(self.df_lagged[self.target], self.df_lagged[self.target], color='red', linestyle='--', label="Perfect Fit")
        plt.xlabel("Actual " + self.target)
        plt.ylabel("Predicted " + self.target)
        plt.title(f"Actual vs Predicted: {self.target}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals(self):
        """Plot residuals of the model."""
        residuals = self.df_lagged[self.target] - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df_lagged[self.target], residuals, color='green', alpha=0.6, label="Residuals")
        plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
        plt.xlabel("Actual " + self.target)
        plt.ylabel("Residuals")
        plt.title(f"Residuals Plot: {self.target}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """Executes the full pipeline: find best lags, create dataset, train model, and plot results."""
        print("Finding best lags...")
        self.find_best_lags()
        print(f"Best lags: {self.best_lags}")

        print("Creating lagged dataset...")
        self.create_lagged_dataset()

        print("Training model...")
        self.train_model()

        # Plotting results
        self.plot_actual_vs_predicted()
        self.plot_residuals()

crisis_list = [2008, 2022]

all_dfs = load_dataset("data", crisis_list)

# crisis_year = 2008
#
# # Loop through each DataFrame in the list
# for i, df in enumerate(all_dfs[crisis_year]):
#     plotting(df)
#
# merged_2008_df = merge_dataframes_on_date(*all_dfs[crisis_year])  # Unpack list and pass it as separate arguments
# merged_2008_df = normalisation(merged_2008_df)
#
# plotting(merged_2008_df)
#
# plot_correlation_matrix(merged_2008_df, 'Correlation Matrix')
#
# granger_causality_2008 = GrangerCausalityAnalyzer(merged_2008_df, maxlag=10)
# granger_causality_2008.run_workflow()
#
# selected_features = lasso_feature_selection(merged_2008_df, "GDP")
#
# # Initialize model
# ts_model = TimeSeriesRegression(merged_2008_df, selected_features, 'GDP', max_lag=6)
#
# # Run the full process
# ts_model.run()
#
# # Access results
# print("Optimal Lags:", ts_model.best_lags)
# print("Model MSE:", ts_model.mse)


target_step_column = 'Cap rate'
annotate_events = {
    "Russia Invades Ukraine": ("2022-02-24", True),
    "Price Cap not implemented \n(as above Energy Price Guarantee)": ("2023-03-01", False)
}
# Define EPG periods
highlight_periods = [
    (pd.to_datetime('2022-10-01'), pd.to_datetime('2023-04-01'), 2500, "Energy Price Guarantee (£2,500)", 'crimson'),
    (pd.to_datetime('2023-04-01'), pd.to_datetime('2023-07-01'), 3000, "Energy Price Guarantee (£3,000)", 'purple')
]
# Loop through each DataFrame in the list for the specific year
for i, df in enumerate(all_dfs[2022]):
    if target_step_column in df.columns:
        plotting(df, step_column=target_step_column, annotation = annotate_events, highlight = highlight_periods)