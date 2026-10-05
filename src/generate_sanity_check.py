import pandas as pd
import numpy as np
import os
import glob
from utils.config import PREDICTIONS_DIR

def main():
    print("Generating sanity check table...")
    s_id = "Uttar_Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ"
    
    # Load Naive
    naive_path = os.path.join(PREDICTIONS_DIR, f"{s_id}_Naive_ExpA_1step_test.csv")
    lstm_path = os.path.join(PREDICTIONS_DIR, f"{s_id}_LSTM_ExpA_1step_test.csv")
    gru_path = os.path.join(PREDICTIONS_DIR, f"{s_id}_GRU_ExpA_1step_test.csv")
    
    if not os.path.exists(naive_path):
        print(f"Missing {naive_path}")
        return
        
    df_naive = pd.read_csv(naive_path)
    df_lstm = pd.read_csv(lstm_path)
    df_gru = pd.read_csv(gru_path)
    
    # We'll take the first 20 test observations
    n_obs = min(20, len(df_naive))
    
    sanity_df = pd.DataFrame({
        'Date': df_naive['Date'].iloc[:n_obs] if 'Date' in df_naive.columns else [f'Test_Day_{i}' for i in range(1, n_obs+1)],
        'Actual': df_naive['Actual'].iloc[:n_obs],
        'Naive Prediction': df_naive['Predicted'].iloc[:n_obs],
        'LSTM Prediction': df_lstm['Predicted'].iloc[:n_obs],
        'GRU Prediction': df_gru['Predicted'].iloc[:n_obs]
    })
    
    sanity_out = os.path.join(PREDICTIONS_DIR, "neural_model_sanity_check.csv")
    sanity_df.to_csv(sanity_out, index=False)
    print(f"Sanity check saved to {sanity_out}")
    
    print("\nSummary Statistics on entire validation set:")
    for name, df in zip(['Actual', 'Naive', 'LSTM', 'GRU'], 
                        [df_naive['Actual'], df_naive['Predicted'], df_lstm['Predicted'], df_gru['Predicted']]):
        print(f"{name}:")
        print(f"  Min: {df.min():.2f}")
        print(f"  Max: {df.max():.2f}")
        print(f"  Mean: {df.mean():.2f}")
        print(f"  Median: {df.median():.2f}")
        print(f"  Std: {df.std():.2f}")
        print()

if __name__ == '__main__':
    main()
