import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data.preprocessing import process_pipeline
from data.feature_engineering import feature_engineering_pipeline

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(base_dir, "Agriculture_price_dataset.csv")
    output_path = os.path.join(base_dir, "data", "processed", "selected_series.csv")
    
    # Based on the analysis, these are the selected series
    # Primary series
    primary_series = "Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ"
    
    # Secondary series
    secondary_series_1 = "West Bengal_paschim bardhaman_Asansol_Potato_Jyoti_FAQ"
    secondary_series_2 = "Uttar Pradesh_sambhal_Sambhal_Potato_Desi_FAQ"
    
    selected_series = [primary_series, secondary_series_1, secondary_series_2]
    
    print("Running Data Pipeline...")
    df_selected = process_pipeline(input_path, output_path, selected_series)
    
    print("\nRunning Feature Engineering Pipeline (Preview)...")
    df_features = feature_engineering_pipeline(df_selected)
    
    features_output_path = os.path.join(base_dir, "data", "processed", "selected_series_features.csv")
    df_features.to_csv(features_output_path, index=False)
    print(f"Features dataset saved to {features_output_path}")

if __name__ == "__main__":
    main()
