import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import stats
import time

def detect_and_remove_outliers(X, y, threshold=3.0):
    # Skip outlier detection for small datasets
    if len(X) < 20:
        return X, y
    
    # Calculate z-scores for the target variable
    mean_y = np.mean(y)
    std_y = np.std(y)
    z_scores = np.abs((y - mean_y) / std_y)
    
    mask = z_scores < threshold
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    return X_clean, y_clean

def convert_to_one_hot(X_train, X_test):
    X_train_df = pd.DataFrame(X_train)
    
    # Initialize DataFrames to store results
    X_train_one_hot = pd.DataFrame(index=X_train_df.index)
    X_test_df = pd.DataFrame(X_test)
    X_test_one_hot = pd.DataFrame(index=X_test_df.index)
    
    # Process each column
    for column in X_train_df.columns:
        # Get all unique categories
        all_categories = set(X_train_df[column].astype(str).unique())
        all_categories.update(X_test_df[column].astype(str).unique())
        
        cat_to_int = {cat: i for i, cat in enumerate(sorted(all_categories))}
        train_cat = X_train_df[column].astype(str).map(cat_to_int)
        test_cat = X_test_df[column].astype(str).map(cat_to_int)
        
        # Create one-hot encoding
        train_one_hot = pd.get_dummies(train_cat, prefix=column)
        test_one_hot = pd.get_dummies(test_cat, prefix=column)
        test_one_hot = test_one_hot[train_one_hot.columns]
        
        # Add to results
        X_train_one_hot = pd.concat([X_train_one_hot, train_one_hot], axis=1)
        X_test_one_hot = pd.concat([X_test_one_hot, test_one_hot], axis=1)
    
    return X_train_one_hot, X_test_one_hot

def create_neural_network(input_dim):
    # Create input layer
    inputs = Input(shape=(input_dim,))
    
    # First layer
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(inputs)
    x = Dropout(0.2)(x)
    
    # Second layer 
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = Dropout(0.2)(x)
    
    # Second layer
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Output layer
    outputs = Dense(1)(x)
    
    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mae')
    
    return model

def evaluate_model(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def calculate_improvement(dl_metrics, lr_metrics):
    improvements = {}
    for metric in dl_metrics:
        # For R2, higher is better
        if metric == 'R2':
            if lr_metrics[metric] > 0:
                improvements[metric] = ((dl_metrics[metric] - lr_metrics[metric]) / abs(lr_metrics[metric])) * 100
            else:
                # If baseline R2 is negative or zero
                improvements[metric] = dl_metrics[metric] - lr_metrics[metric]
        else:
            # For error metrics, lower is better
            improvements[metric] = ((lr_metrics[metric] - dl_metrics[metric]) / lr_metrics[metric]) * 100
            
    return improvements

def run_statistical_tests(method1_values, method2_values, alternative='two-sided'):
    method1_values = np.array(method1_values)
    method2_values = np.array(method2_values)
    differences = method1_values - method2_values
    
    # Paired test
    statistic, p_value = stats.wilcoxon(method1_values, method2_values, alternative=alternative)
    significant = p_value < 0.05
    
    # Calculate effect size
    effect_size = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
    
    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": significant,
        "effect_size": effect_size,
    }

def save_results_to_csv(summary_df, all_metric_values, output_file='results/data/all_results.csv'):
    results_df = summary_df.copy()
    
    # Add standard deviation columns
    for metric in ['MAPE', 'MAE', 'RMSE', 'R2']:
        results_df[f'LR_{metric}_std'] = np.nan
        results_df[f'DL_{metric}_std'] = np.nan
    
    # Fill in standard deviations
    for i, row in results_df.iterrows():
        system_key = f"{row['System']}_{row['Dataset']}"
        if system_key in all_metric_values:
            for metric in ['MAPE', 'MAE', 'RMSE', 'R2']:
                lr_values = all_metric_values[system_key][f'LR_{metric}']
                dl_values = all_metric_values[system_key][f'DL_{metric}']
                
                results_df.at[i, f'LR_{metric}_std'] = np.std(lr_values)
                results_df.at[i, f'DL_{metric}_std'] = np.std(dl_values)
    
    # Add significance columns
    for metric in ['MAPE', 'MAE', 'RMSE', 'R2']:
        results_df[f'{metric}_significant'] = results_df[f'{metric}_p_value'] < 0.05
    
    # Save aggregated results to CSV
    results_df.to_csv(output_file, index=False)
    
    # CSV with individual repeat results
    detailed_rows = []
    
    for i, row in summary_df.iterrows():
        system_key = f"{row['System']}_{row['Dataset']}"
        if system_key in all_metric_values:
            num_repeats = len(all_metric_values[system_key]['LR_MAPE'])
            
            for repeat_idx in range(num_repeats):
                detailed_row = {
                    'System': row['System'],
                    'Dataset': row['Dataset'],
                    'Repeat': repeat_idx + 1
                }
                
                for metric in ['MAPE', 'MAE', 'RMSE', 'R2']:
                    if f'LR_{metric}' in all_metric_values[system_key]:
                        detailed_row[f'LR_{metric}'] = all_metric_values[system_key][f'LR_{metric}'][repeat_idx]
                    
                    if f'DL_{metric}' in all_metric_values[system_key]:
                        detailed_row[f'DL_{metric}'] = all_metric_values[system_key][f'DL_{metric}'][repeat_idx]
                
                detailed_rows.append(detailed_row)
    
    # Create and save the detailed DataFrame
    detailed_output_file = output_file.replace('.csv', '_detailed.csv')
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(detailed_output_file, index=False)
    
    return output_file, detailed_output_file

def main():
    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    # systems = ['dconvert']  # Uncomment to use only one system for testing
    num_repeats = 10
    train_frac = 0.7
    random_seed = 1 

    # Neural network training parameters
    epochs = 200
    batch_size = 32
    patience = 10
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    
    # Summary results across all systems
    summary_results = {
        'System': [],
        'Dataset': [],
        'DL_MAPE': [], 'LR_MAPE': [], 'MAPE_Improv': [], 'MAPE_p_value': [],
        'DL_MAE': [], 'LR_MAE': [], 'MAE_Improv': [], 'MAE_p_value': [],
        'DL_RMSE': [], 'LR_RMSE': [], 'RMSE_Improv': [], 'RMSE_p_value': [],
        'DL_R2': [], 'LR_R2': [], 'R2_Improv': [], 'R2_p_value': []
    }

    # Store raw metric values
    all_metric_values = {}

    # Store training times
    training_times = {
        'System': [],
        'Dataset': [],
        'LR_Time': [],
        'DL_Time': []
    }

    for current_system in systems:
        print(f"\n{'='*50}")
        print(f"Processing system: {current_system}")
        print(f"{'='*50}")
        
        datasets_location = f'datasets/{current_system}'

        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]

        for csv_file in csv_files:
            system_key = f"{current_system}_{csv_file}"
            all_metric_values[system_key] = {
                'DL_MAPE': [], 'LR_MAPE': [],
                'DL_MAE': [], 'LR_MAE': [],
                'DL_RMSE': [], 'LR_RMSE': [],
                'DL_R2': [], 'LR_R2': []
            }
            
            print(f"\n> System: {current_system}, Dataset: {csv_file}")
            print(f"> Training data fraction: {train_frac}, Number of repeats: {num_repeats}")

            data = pd.read_csv(os.path.join(datasets_location, csv_file))

            # For repeated evaluations
            dl_metrics = {'MAPE': [], 'MAE': [], 'RMSE': [], 'R2': []}
            lr_metrics = {'MAPE': [], 'MAE': [], 'RMSE': [], 'R2': []}
            
            # Training times
            lr_times = []
            dl_times = []

            for current_repeat in range(num_repeats):
                print(f"\nRepeat {current_repeat + 1}/{num_repeats}")
                
                # Use different random seed for each repeat
                current_seed = random_seed * (current_repeat + 1)
                
                # Randomly split data into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=current_seed)
                test_data = data.drop(train_data.index)

                # Split features and target
                X_train = train_data.iloc[:, :-1]
                y_train = train_data.iloc[:, -1]
                X_test = test_data.iloc[:, :-1]
                y_test = test_data.iloc[:, -1]
                
                # Linear Regression
                lr_start_time = time.time()
                
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_predictions = lr_model.predict(X_test)
                
                lr_end_time = time.time()
                lr_time = lr_end_time - lr_start_time
                lr_times.append(lr_time)
                
                # Evaluate linear regression model
                lr_results = evaluate_model(y_test, lr_predictions)
                for metric, value in lr_results.items():
                    lr_metrics[metric].append(value)
                    all_metric_values[system_key][f'LR_{metric}'].append(value)

                # Deep Learning Model
                dl_start_time = time.time()
                
                # 1. Outlier
                X_train, y_train = detect_and_remove_outliers(X_train, y_train)

                # 2. One hot encoding
                X_train_onehot, X_test_onehot = convert_to_one_hot(X_train, X_test)

                # 3. Feature selection
                    # Calculate feature importance on one-hot encoded data
                mi_scores = mutual_info_regression(X_train_onehot, y_train)
                
                    # Create a list of (index, score) tuples
                feature_scores = [(i, mi_scores[i]) for i in range(len(mi_scores))]
                
                    # Sort by descending importance
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                    # Calculate total importance
                total_importance = sum([score for _, score in feature_scores])
                
                    # Select features until we reach the threshold
                selected_onehot_indices = []
                cumulative_importance = 0
                
                for idx, score in feature_scores:
                    selected_onehot_indices.append(idx)
                    cumulative_importance += score / total_importance
                    
                    if cumulative_importance >= 0.95:
                        break
                
                    # Ensure at least one feature is selected
                if not selected_onehot_indices:
                    selected_onehot_indices = [feature_scores[0][0]]
                
                    # Sort indices
                selected_onehot_indices.sort()

                # Use selected indices
                X_train_selected = X_train_onehot.iloc[:, selected_onehot_indices]
                X_test_selected = X_test_onehot.iloc[:, selected_onehot_indices]

                # Create and train the neural network
                model = create_neural_network(X_train_selected.shape[1])
                
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=0.0001
                    )
                ]
                
                # Train the model
                model.fit(
                    X_train_selected, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )

                # Make predictions
                dl_predictions = model.predict(X_test_selected).flatten()
                
                dl_end_time = time.time()
                dl_time = dl_end_time - dl_start_time
                dl_times.append(dl_time)

                # Evaluate deep learning model
                dl_results = evaluate_model(y_test, dl_predictions)
                for metric, value in dl_results.items():
                    dl_metrics[metric].append(value)
                    all_metric_values[system_key][f'DL_{metric}'].append(value)
                
                print(f"LR - MAPE: {lr_results['MAPE']:.4f}, MAE: {lr_results['MAE']:.4f}, RMSE: {lr_results['RMSE']:.4f}, R2: {lr_results['R2']:.4f}")
                print(f"DL - MAPE: {dl_results['MAPE']:.4f}, MAE: {dl_results['MAE']:.4f}, RMSE: {dl_results['RMSE']:.4f}, R2: {dl_results['R2']:.4f}")

            # Calculate average metrics and standard deviations across all repeats
            avg_dl_metrics = {metric: np.mean(values) for metric, values in dl_metrics.items()}
            std_dl_metrics = {metric: np.std(values) for metric, values in dl_metrics.items()}
            
            avg_lr_metrics = {metric: np.mean(values) for metric, values in lr_metrics.items()}
            std_lr_metrics = {metric: np.std(values) for metric, values in lr_metrics.items()}
            
            # Calculate average training times
            avg_lr_time = np.mean(lr_times)
            avg_dl_time = np.mean(dl_times)
            
            # Store training times
            training_times['System'].append(current_system)
            training_times['Dataset'].append(csv_file)
            training_times['LR_Time'].append(avg_lr_time)
            training_times['DL_Time'].append(avg_dl_time)
            
            # Calculate improvements
            improvements = calculate_improvement(avg_dl_metrics, avg_lr_metrics)
            
            # Run statistical tests
            p_values = {}
            for metric in avg_dl_metrics:
                dl_values = all_metric_values[system_key][f'DL_{metric}']
                lr_values = all_metric_values[system_key][f'LR_{metric}']
                
                if metric != 'R2':
                    # For error metrics, lower is better, so we're testing if DL is lower than LR
                    test_results = run_statistical_tests(dl_values, lr_values)
                else:
                    # For R2, higher is better, so we're testing if DL is higher than LR
                    test_results = run_statistical_tests(lr_values, dl_values)
                
                p_values[metric] = test_results['p_value']
            
            # Add to summary results
            summary_results['System'].append(current_system)
            summary_results['Dataset'].append(csv_file)
            summary_results['DL_MAPE'].append(avg_dl_metrics['MAPE'])
            summary_results['LR_MAPE'].append(avg_lr_metrics['MAPE'])
            summary_results['MAPE_Improv'].append(improvements['MAPE'])
            summary_results['MAPE_p_value'].append(p_values['MAPE'])
            summary_results['DL_MAE'].append(avg_dl_metrics['MAE'])
            summary_results['LR_MAE'].append(avg_lr_metrics['MAE'])
            summary_results['MAE_Improv'].append(improvements['MAE'])
            summary_results['MAE_p_value'].append(p_values['MAE'])
            summary_results['DL_RMSE'].append(avg_dl_metrics['RMSE'])
            summary_results['LR_RMSE'].append(avg_lr_metrics['RMSE'])
            summary_results['RMSE_Improv'].append(improvements['RMSE'])
            summary_results['RMSE_p_value'].append(p_values['RMSE'])
            summary_results['DL_R2'].append(avg_dl_metrics['R2'])
            summary_results['LR_R2'].append(avg_lr_metrics['R2'])
            summary_results['R2_Improv'].append(improvements['R2'])
            summary_results['R2_p_value'].append(p_values['R2'])

    # Save all results to CSV file
    summary_df = pd.DataFrame(summary_results)
    save_results_to_csv(summary_df, all_metric_values)
    pd.DataFrame(training_times).to_csv('results/data/training_times.csv', index=False)

    # Calculate overall average metrics across all systems
    print("\n===== OVERALL AVERAGE METRICS =====")

    print("\nLinear Regression (Overall Average):")
    print(f"Average MAPE: {np.mean(summary_df['LR_MAPE']):.4f}")
    print(f"Average MAE: {np.mean(summary_df['LR_MAE']):.4f}")
    print(f"Average RMSE: {np.mean(summary_df['LR_RMSE']):.4f}")
    print(f"Average R2: {np.mean(summary_df['LR_R2']):.4f}")

    print("\nDeep Learning (Overall Average):")
    print(f"Average MAPE: {np.mean(summary_df['DL_MAPE']):.4f}")
    print(f"Average MAE: {np.mean(summary_df['DL_MAE']):.4f}")
    print(f"Average RMSE: {np.mean(summary_df['DL_RMSE']):.4f}")
    print(f"Average R2: {np.mean(summary_df['DL_R2']):.4f}")

    print("\nAverage Improvements Across All Systems:")
    print(f"MAPE Improvement: {np.mean(summary_df['MAPE_Improv']):.2f}%")
    print(f"MAE Improvement: {np.mean(summary_df['MAE_Improv']):.2f}%")
    print(f"RMSE Improvement: {np.mean(summary_df['RMSE_Improv']):.2f}%")
    print(f"R2 Improvement: {np.mean(summary_df['R2_Improv']):.2f}%")

    # Number of systems where DL outperformed LR
    better_count = {
        'MAPE': sum(summary_df['MAPE_Improv'] > 0),
        'MAE': sum(summary_df['MAE_Improv'] > 0),
        'RMSE': sum(summary_df['RMSE_Improv'] > 0),
        'R2': sum(summary_df['R2_Improv'] > 0)
    }

    total_systems = len(summary_df)

    print("\nDL outperformed LR in:")
    print(f"MAPE: {better_count['MAPE']}/{total_systems} systems ({better_count['MAPE']/total_systems*100:.1f}%)")
    print(f"MAE: {better_count['MAE']}/{total_systems} systems ({better_count['MAE']/total_systems*100:.1f}%)")
    print(f"RMSE: {better_count['RMSE']}/{total_systems} systems ({better_count['RMSE']/total_systems*100:.1f}%)")
    print(f"R2: {better_count['R2']}/{total_systems} systems ({better_count['R2']/total_systems*100:.1f}%)")

    # Statistically significant improvements
    sig_better_count = {
        'MAPE': sum((summary_df['MAPE_Improv'] > 0) & (summary_df['MAPE_p_value'] < 0.05)),
        'MAE': sum((summary_df['MAE_Improv'] > 0) & (summary_df['MAE_p_value'] < 0.05)),
        'RMSE': sum((summary_df['RMSE_Improv'] > 0) & (summary_df['RMSE_p_value'] < 0.05)),
        'R2': sum((summary_df['R2_Improv'] > 0) & (summary_df['R2_p_value'] < 0.05))
    }

    print("\nDL significantly outperformed LR (p < 0.05) in:")
    print(f"MAPE: {sig_better_count['MAPE']}/{total_systems} systems ({sig_better_count['MAPE']/total_systems*100:.1f}%)")
    print(f"MAE: {sig_better_count['MAE']}/{total_systems} systems ({sig_better_count['MAE']/total_systems*100:.1f}%)")
    print(f"RMSE: {sig_better_count['RMSE']}/{total_systems} systems ({sig_better_count['RMSE']/total_systems*100:.1f}%)")
    print(f"R2: {sig_better_count['R2']}/{total_systems} systems ({sig_better_count['R2']/total_systems*100:.1f}%)")

if __name__ == "__main__":
    main()