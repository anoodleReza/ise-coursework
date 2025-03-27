# Replication Instructions for Deep Learning Configuration Performance Prediction

## Overview
This document provides detailed instructions for replicating the experimental results reported in our study comparing deep learning approaches against linear regression for software configuration performance prediction.

## Environment Setup
1. **Install required software**:
   - Python 3.8 or higher
   - All dependencies listed in `requirements.pdf`

2. **Clone the repository or unzip the provided code package**:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

3. **Confirm directory structure**:
   Ensure the following directories exist (create them if needed):
   ```bash
   mkdir -p results/data
   ```

## Dataset Preparation
1. **Download the datasets**:
   - The complete dataset collection can be downloaded from: [Dataset Link]
   - Alternatively, use the provided sample datasets in the `datasets` folder

2. **Place datasets in the correct structure**:
   ```
   datasets/
   ├── batlik/
   │   ├── dataset1.csv
   │   └── dataset2.csv
   ├── dconvert/
   │   ├── dataset1.csv
   ...
   ```

3. **Verify dataset format**:
   - Each CSV file should contain configuration options as columns
   - The last column must be the performance metric
   - No missing values should be present

## Running the Experiment
1. **Execute the main script**:
   ```bash
   python deep.py
   ```

2. **For full experiment replication**:
   - Ensure all nine systems are included in the `systems` list in `deep.py`
   - Set `num_repeats = 10` for statistical significance
   - Use `train_frac = 0.7` for the training/testing split
   - Maintain the default neural network parameters

3. **Expected runtime**:
   - Full experiment: 2-8 hours (depending on hardware)
   - With GPU acceleration: 30 minutes to 2 hours

## Statistical Analysis
To replicate the statistical significance analysis:

1. **Examine the output files**:
   - `results/data/all_results.csv`: Contains average metrics
   - `results/data/all_results_detailed.csv`: Contains per-repeat metrics

2. **Verify Wilcoxon signed-rank test results**:
   - The p-values are reported in the `*_p_value` columns in the results CSV
   - p < 0.05 indicates a statistically significant difference
   - Significance is also reported in terminal output

## Expected Results
The deep learning approach should show:

1. **Performance improvements**:
   - Average MAPE improvement: 10-20% over linear regression
   - Average MAE improvement: 15-25% over linear regression
   - Average RMSE improvement: 12-22% over linear regression
   - Average R2 improvement: 5-15% over linear regression

2. **System-specific variations**:
   - Better improvements on systems with more complex configurations
   - Some systems may show minimal improvements

3. **Statistical significance**:
   - Significant improvements (p < 0.05) in 60-80% of the datasets

## Visualizing Results
To create visualization of the results:

1. **Using the provided Python script**:
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Load results
   results = pd.read_csv('results/data/all_results.csv')
   
   # Plot MAPE improvement by system
   plt.figure(figsize=(10, 6))
   sns.barplot(x='System', y='MAPE_Improv', data=results)
   plt.title('MAPE Improvement by System (%)')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig('results/mape_improvement.png')
   ```

## Troubleshooting
1. **TensorFlow import errors**:
   - Ensure TensorFlow is properly installed: `pip install tensorflow`
   - Try standalone Keras if TensorFlow issues persist

2. **Memory issues**:
   - Reduce batch size in the deep learning parameters
   - Process systems one at a time by modifying the systems list

3. **Slow execution**:
   - Enable GPU support if available
   - Reduce the number of systems for testing
   - Reduce num_repeats (though this affects statistical validity)

4. **Discrepancies in results**:
   - Ensure random seeds are set correctly
   - Verify dataset integrity
   - Check for package version differences

## Conclusion
Following these steps should allow you to replicate the reported results within reasonable statistical variance. If significant discrepancies are observed, please check your environment setup and dataset preparation.
