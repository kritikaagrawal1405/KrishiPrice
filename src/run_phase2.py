import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.feature_engineering import apply_feature_engineering
from forecasting.forecast_pipeline import ForecastPipeline
from utils.config import CONFIG, DATA_DIR, RESULTS_DIR, PLOTS_DIR
from utils.plotting import plot_history, plot_train_val_test

def main():
    print("Loading Phase 1 processed data...")
    selected_series_path = os.path.join(DATA_DIR, 'selected_series.csv')
    df = pd.read_csv(selected_series_path)
    
    print("Applying Feature Engineering...")
    df_features = apply_feature_engineering(df, CONFIG)
    
    all_results = []
    
    # Generate overall plots for primary series
    s_id = CONFIG['primary_series']
    df_primary = df_features[df_features['Series_ID'] == s_id]
    plot_history(df_primary, s_id.split('_')[-1], PLOTS_DIR)
    
    # Train / Val / Test split plotting
    n = len(df_primary)
    train_end = int(n * CONFIG['train_ratio'])
    val_end = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))
    plot_train_val_test(df_primary.iloc[:train_end], df_primary.iloc[train_end:val_end], df_primary.iloc[val_end:], s_id.split('_')[-1], PLOTS_DIR)

    
    print("\n--- Running on PRIMARY SERIES ---")
    s_id = CONFIG['primary_series']
    
    for horizon in CONFIG['horizons']:
        print(f"\nHorizon: {horizon}-step ahead")
        
        # Experiment A
        print("Running Experiment A (Univariate)...")
        pipeline_A = ForecastPipeline(df_features, s_id, horizon=horizon, experiment='A')
        res_A = pipeline_A.execute()
        all_results.append(res_A)
        
        # Experiment B
        print("Running Experiment B (Feature-enhanced)...")
        pipeline_B = ForecastPipeline(df_features, s_id, horizon=horizon, experiment='B')
        res_B = pipeline_B.execute()
        all_results.append(res_B)
        
    print("\n--- Running Generalization on SECONDARY SERIES ---")
    # For secondary series, we can just test the best config, but for now we'll run XGBoost (Exp B) as a generalization test
    for s_id in CONFIG['secondary_series']:
        print(f"\nSeries: {s_id}")
        for horizon in CONFIG['horizons']:
            pipeline_gen = ForecastPipeline(df_features, s_id, horizon=horizon, experiment='B')
            pipeline_gen._prepare_data()
            pipeline_gen.run_baselines() # Baseline for reference
            pipeline_gen.run_xgboost()
            res_gen = pd.DataFrame(pipeline_gen.results)
            res_gen['Series'] = s_id
            all_results.append(res_gen)
            
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, 'metrics', 'model_comparison.csv')
    final_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    print("\nFinal Model Comparison:")
    print(final_results.to_string())

if __name__ == "__main__":
    main()
