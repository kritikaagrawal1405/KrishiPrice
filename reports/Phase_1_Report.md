# KrishiPrice - Phase 1 Report: Data Pipeline & Problem Definition (Corrected)

## 1. Dataset Overview
- **Number of records**: 737,392
- **Number of columns**: 10 (`STATE`, `District Name`, `Market Name`, `Commodity`, `Variety`, `Grade`, `Min_Price`, `Max_Price`, `Modal_Price`, `Price Date`)
- **Missing values**: None (0.0% across all columns)
- **Unique categories**: 30 States, 373 Districts, 1598 Markets, 5 Commodities, 94 Varieties, 7 Grades.
- **Global Date range**: 2023-06-06 to 2025-06-11 (737 unique dates)

## 2. Data-Quality Findings
- **Target Invalidity (Modal_Price <= 0)**: 3 rows.
- **Feature Invalidity (Min/Max <= 0)**: 678 rows. 
- **Range Violations**: 1860 rows (0.25% of dataset) where `Min_Price > Max_Price` or `Modal_Price` is outside `[Min_Price, Max_Price]`.
- **Duplicate Aggregation**: An exact analysis of the dataset verified that there are exactly 0 duplicate observations at the (Series_ID + Price Date) level. Therefore, there are no intra-day duplicates requiring aggregation. 

## 3. Correct Temporal Structure
- The dataset contains observations across 737 unique calendar dates. 
- **Important**: Individual market–commodity series have different observation frequencies and temporal coverage. The dataset is NOT one global daily time series.
- The selected primary series is approximately daily, but retains explicit observation dates.

## 4. Series Analysis
A unique series was defined as a unique combination of `STATE + District + Market + Commodity + Variety + Grade`.
- Many combinations appear infrequently. Meaningful forecasting requires selecting series with a continuous and long history.
- The longest series in the dataset span ~728 observed days out of a possible 737 calendar days.

## 5. Primary Series
**Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ**
- **Observations**: 728
- **Date Range**: 2023-06-06 to 2025-06-11
- **Expected calendar days**: 737
- **Observed days**: 728
- **Coverage percentage**: 98.78%
- **Median gap**: 1.0 days
- **Maximum gap**: 3.0 days
- **Number of missing calendar days**: 9
- **Number of gaps > 1 day**: 8
- **Conclusion**: Retained as the primary prototype series due to exceptional coverage and continuity.

## 6. Secondary Series
1. **West Bengal_paschim bardhaman_Asansol_Potato_Jyoti_FAQ**
   - **Observations**: 720
   - **Date Range**: 2023-06-06 to 2025-06-11
   - **Coverage percentage**: 97.69%
   - **Missing/Gap**: Median gap 1.0 days, maximum gap 3.0 days.
2. **Uttar Pradesh_sambhal_Sambhal_Potato_Desi_FAQ**
   - **Observations**: 728
   - **Date Range**: 2023-06-06 to 2025-06-11
   - **Coverage percentage**: 98.78%
   - **Missing/Gap**: Median gap 1.0 days, maximum gap 4.0 days.

## 7. Corrected Data-Cleaning Policy
- **Target Invalidity**: If `Modal_Price <= 0`, the observation is not suitable as a target and is removed from the forecasting dataset.
- **Feature Invalidity**: If `Min_Price` or `Max_Price` is invalid (<= 0) but `Modal_Price` is valid, the observation is *retained*. Problematic Min/Max values are flagged/marked invalid and excluded from feature engineering unless explicitly handled later.
- **Range Violations**: For `Min_Price > Max_Price` or `Modal_Price` out of bounds, we do not automatically assume the entire observation is useless. These records are flagged (`range_violation=True`). Phase 2 models will prioritize the validity of `Modal_Price` and treat the bounding features as conditionally valid. No silent deletions.

## 8. Corrected Missing-Date Policy
- Missing observations are not treated as zero-price observations and are not automatically replaced with interpolated prices.
- **Policy**: 
  1. Preserve actual observed prices.
  2. Do not automatically interpolate `Modal_Price`.
  3. Do not automatically forward-fill `Modal_Price`.
  4. Keep the raw/clean observed timeline intact.
  5. If Phase 2 requires a regular daily sequence for a specific model, missing dates must be handled explicitly in Phase 2.
  6. Any imputation, if eventually required, must be limited to clearly defined short gaps, documented, performed without leakage, marked with an `is_imputed` indicator, and evaluated against the non-imputed approach.

## 9. Outlier Policy
- Agricultural price spikes often correspond to genuine market events.
- We do not automatically remove outliers. We apply robust bounds to *flag* extreme deviations. Flagged outliers will be preserved in the training dataset, as capturing these dynamics is essential for a robust forecasting engine.

## 10. Feature-Engineering Plan
*Note: The final Phase 2 training matrix is not created here. Features will be generated in Phase 2 using only past information.*
- **Time Features**: `day_of_week`, `month`, `week_of_year`, cyclical encodings.
- **Lag Features**: Lags of 1, 3, 7, 14, and 30 days.
- **Rolling Statistics**: Rolling mean, standard deviation, min, and max.
- **Price Features**: `price_spread` and `spread_ratio`.

## 11. Forecast Horizon Definition
Given the 98%+ coverage of the primary series, Phase 2 will interpret the horizons in observation steps (which heavily overlap with calendar days):
- **1-step ahead**: "next available observation".
- **7-step ahead**: "7th future observation".
- *Note*: It is not automatically assumed that 7-step ahead means exactly 7 calendar days due to the 9 missing calendar days in the timeline.

## 12. Chronological Train/Validation/Test Strategy
- **Split Mechanism**: Chronological Split.
- **Proportions**: 
  - **Train**: First 70% of sequential data.
  - **Validation**: Next 15%.
  - **Test**: Final 15%.

## 13. Leakage Prevention Rules
The following strict rules apply to Phase 2:
1. **Chronological split**: Train/validation/test split must strictly follow chronological order.
2. **No random shuffling**: Time series sequence must not be broken during dataset splits.
3. **Scaler fitting**: Scalers are fitted only on training data.
4. **Rolling features causality**: Rolling features must use only information available before the prediction timestamp.
5. **Lag causality**: Lag features must only use past observations.
6. **No target leakage**: Future target values must never appear as input features.
7. **Imputation causality**: Any imputation strategy must be fitted/applied without using future information.
8. **No evaluation leakage**: Feature engineering must not allow validation/test information to influence training.
9. **Feature availability**: If `Min_Price` and `Max_Price` are used to predict `t+1`, only values available at `t` or earlier may be used.
10. **Same-time variables restriction**: Same-time variables must not be used if they would only become available after the prediction is supposed to be made.

Explicit Rule for Prediction at time `t`:
- **Available information**: `t` and earlier
- **Target**: `t+1` or future

## 14. Phase 2 Input Definition
For Phase 2, we define two possible experimental setups:
- **EXPERIMENT A**: Univariate forecasting. Input: Historical `Modal_Price` only.
- **EXPERIMENT B**: Feature-enhanced time-series forecasting. Input may include Historical `Modal_Price`, lag features, rolling features, calendar/seasonality features, and `Min_Price` / `Max_Price` *only* when they are legitimately available at prediction time.

*Both experiments share the same Target: Future `Modal_Price`.*

## 15. Ready for Phase 2

**PHASE 1 COMPLETE**

- [x] Dataset audited
- [x] Temporal structure verified
- [x] Primary series verified
- [x] Secondary series verified
- [x] Cleaning policy finalized
- [x] Missing-date policy finalized
- [x] Duplicate policy verified
- [x] Feature plan finalized
- [x] Forecast horizon defined
- [x] Leakage rules finalized
- [x] Phase 2 inputs defined

Phase 2 can now begin.
