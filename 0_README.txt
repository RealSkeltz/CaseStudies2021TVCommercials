STRUCTURE OF CODE

1_data_initialization.py
Input: broadcasting_data, traffic_data
Goal: Filter on relevant visits and commercials, create/engineer additional features
Output: broadcast_incl_feat.csv, broadcast_relevant_NL.csv, broadcast_for_models.csv, traffic_relevant_NL.csv

2_data_initialization_ARIMA.py
Input: broadcasting_data, traffic_data
Goal: Create data suitable for ARIMA model: visits aggregated on minutes and features of ads 1 min ago
Output: data_for_ARIMA.csv

3_data_exploration.py
Input: broadcast_incl_feat.csv, traffic_data, traffic_relevant_NL.csv, broadcast_relevant_NL.csv
Goal: Exploratory plots and descriptive tables for data description section
Output: Report; Tables 1-4, 13-16 and Figures 2-6

4_determine_prepost_effect_visuals.py
Input: broadcast_for_models.csv, traffic_relevant_NL.csv
Goal: Determine length of prepost window and cutoff points
Output: Report; Figures 7-9

5_sum_stats_effect_variable.py
Input: broadcasting_data, traffic_data
Goal: Compute the summary statistics of the prepost windows of 3 and 5 minutes (before excluding overlapping commercials)
Output: Report; Table 6

6_plots_outliers_at_00h.py
Input: broadcast_relevant_NL.csv, traffic_relevant_NL.csv
Goal: Discover how to deal with outliers at 0.00h
Output: Report; Figure 10, Appendix Section 9.1.5

7_classifier_comparison.py
Input: broadcast_for_models.csv
Goal: Selection of classifier method
Output: Report; Table 7

8_classifier_forward_selection.py
Input: broadcast_for_models.csv
Goal: Forward feature selection of AdaBoost with Decision Tree on three different ways
Output: Report; Table 8

9_plots_classifier_forward_selection.R
Input: values of forward feature selection 7_classifier_forward_selection.py
Goal: Plot the values of forward selection of recall and accuracy
Output: Report; Figures 15 and 16

10_sensitivity_analysis.py
Input: broadcast_for_models.csv
Goal: Perform sensitivity analysis for cutoff point of capped effect
Output: Report; Table 9

11_tobit_forward_feature_selection.R
Input: broadcast_for_models.csv
Goal: Forward feature selection of Tobit and fit resulting model
Output: Report; Figure 17, Table 10

12_tobit_assumptions.R
Input: broadcast_for_models.csv
Goal: Verify homoskedasticity and normality assumptions of Tobit
Output: Report; Figures 18 and 19

13_tobit_linearity_assumption.py
Input: broadcast_for_models.csv
Goal: Verify the linearity assumption of Tobit
Output: Report; Table 11

14_ARIMA_analysis.py
Input: data_for_ARIMA.csv
Goal: Perform ARIMA analysis
Output: Report; Tables 12 and 22-23, Figures 13 and 20-23