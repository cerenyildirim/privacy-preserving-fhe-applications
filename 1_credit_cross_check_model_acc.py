import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.datasets import fetch_credit_card
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# --- Configuration ---
N_RUNS = 10   # Number of times to run the experiment with different data shuffles
FINETUNE_SPLIT_FRAC = 0.90
BASE_LR = 0.01       # learning rate for base training
FINETUNE_LR = 0.05  # smaller learning rate for fine-tuning

# --- Helper: decision boundary selector ---
def get_best_split(feature, label):
    """Fit a decision stump to find the threshold best separating feature wrt label."""
    X = feature.values.reshape(-1, 1)
    y = label.values
    tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    tree.fit(X, y)
    threshold = tree.tree_.threshold[0]
    return threshold

# --- Helper: build Keras model ---
def build_model(input_shape, lr):
    """Builds and compiles a new Keras model with given learning rate."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='softplus', use_bias=False),
        tf.keras.layers.Dense(32, activation='softplus', use_bias=False),
        tf.keras.layers.Dense(16, activation='softplus', use_bias=False),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Store results from all runs ---
results = []

for i in range(N_RUNS):
    print(f"\n{'='*30} RUN {i + 1}/{N_RUNS} {'='*30}")

    # Load data
    X, y = fetch_credit_card(return_X_y=True, as_frame=True)
    y = y.map({'0': 0, '1': 1})

    # Find best feature threshold using decision trees
    # features = [f'x{i}' for i in range(1, 9)]

    # Compute and print thresholds for all features
    # thresholds = {}
    # for feat in features:
    #     thresh = get_best_split(X[feat], y)
    #     thresholds[feat] = thresh
    #     print(f"Auto-detected threshold for {feat}: {thresh:.3f}")

    # Create Train/Validation/Test splits
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, train_size=0.7, random_state=i, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=i, stratify=y_test_val 
    )

    # Feature selection using Random Forest on training set 
    # rf = RandomForestClassifier(random_state=42, n_estimators=200)
    # rf.fit(X_train, y_train)

    # importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    # top16_features = importances.sort_values(ascending=False).head(16).index.tolist()

    # print("Top 16 features (selected on training):", top16_features)

    # Apply the same feature selection everywhere
    # X_train = X_train[top16_features]
    # X_val   = X_val[top16_features]
    # X_test  = X_test[top16_features]

    # Specialized datasets for fine-tuning
    selected_indices_train = (
        (X_train['x6'] > 1.5) | 
        (X_train['x5'] < 25.5) 
    )
    X_train_edu_all = X_train[selected_indices_train]
    y_train_edu_all = y_train[selected_indices_train]

    X_train_finetune, X_train_remain, y_train_finetune, y_train_remain = train_test_split(
        X_train_edu_all, y_train_edu_all, train_size=FINETUNE_SPLIT_FRAC, random_state=42
    )
    X_train_base = pd.concat([X_train[~selected_indices_train], X_train_remain])
    y_train_base = pd.concat([y_train[~selected_indices_train], y_train_remain])

    print("Length of the base set:", len(X_train_base))
    print("Length of the finetuning set:", len(X_train_finetune))

    # Scale features based on the validation set (which just an auxiliary set here)
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    X_train_base_scaled = scaler.transform(X_train_base)
    X_train_finetune_scaled = scaler.transform(X_train_finetune)
    X_test_scaled = scaler.transform(X_test)

    # Train base model
    print("\n--- Training Base Model ---")
    base_model = build_model(X_train_base_scaled.shape[1], BASE_LR)
    base_model.fit(
        X_train_base_scaled, y_train_base,
        validation_data=(X_val_scaled, y_val),
        epochs=10,
        batch_size=16,
        verbose=0
    )

    # Evaluate base model
    y_pred_base = (base_model.predict(X_test_scaled).ravel() > 0.5).astype(int)
    acc_base_all = accuracy_score(y_test, y_pred_base)
    selected_indices_test = (
        (X_test['x6'] > 1.5) | 
        (X_test['x5'] < 25.5) 
    )
    mask_test = selected_indices_test.to_numpy()
    y_pred_base_sel = y_pred_base[mask_test]
    acc_base_sel = accuracy_score(y_test[mask_test], y_pred_base_sel)

    print(f"Base Accuracy (All): {acc_base_all:.4f} | Base Accuracy (Selected): {acc_base_sel:.4f}")

    # Fine-tuning
    print("\n--- Fine-tuning Model ---")
    for layer in base_model.layers[:-1]:
        layer.trainable = False
        
    base_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    base_model.fit(
        X_train_finetune_scaled[:2000], y_train_finetune[:2000],
        batch_size=16,
        epochs=1,
        verbose=0,
        validation_data=(X_val_scaled, y_val)
    )

    # Evaluate fine-tuned model
    y_pred_ft = (base_model.predict(X_test_scaled).ravel() > 0.5).astype(int)
    acc_ft_all = accuracy_score(y_test, y_pred_ft)
    y_pred_ft_sel = y_pred_ft[mask_test]

    acc_ft_sel = accuracy_score(y_test[mask_test], y_pred_ft_sel)

    print(f"Fine-tuned Accuracy (All): {acc_ft_all:.4f} | Fine-tuned Accuracy (Selected): {acc_ft_sel:.4f}")

    # Store results
    results.append({
        'run': i + 1,
        'acc_base_all': acc_base_all,
        'acc_base_sel': acc_base_sel,
        'acc_ft_all': acc_ft_all,
        'acc_ft_sel': acc_ft_sel
    })
    # Majority class baseline
    chance_acc_all = max(y_test.value_counts()) / len(y_test)
    chance_acc_selected = max(y_test[mask_test].value_counts()) / len(y_test[mask_test])

    print(f"Chance-level Accuracy (All): {chance_acc_all:.4f}")
    print(f"Chance-level Accuracy (Selected): {chance_acc_selected:.4f}")


# --- Final Analysis ---
print(f"\n{'='*30} FINAL RESULTS (after {N_RUNS} runs) {'='*30}")
results_df = pd.DataFrame(results)

# Calculate improvements
results_df['improvement_all'] = results_df['acc_ft_all'] - results_df['acc_base_all']
results_df['improvement_sel'] = results_df['acc_ft_sel'] - results_df['acc_base_sel']

summary = results_df.agg(['mean', 'std'])
print(summary[['acc_base_all', 'acc_ft_all', 'improvement_all',
               'acc_base_sel', 'acc_ft_sel', 'improvement_sel']].round(4))

# Pretty print summary
avg_improvement_sel = summary.loc['mean', 'improvement_sel']
print("\nSummary:")
if avg_improvement_sel > 0:
    print(f"✅ On average, fine-tuning improved the accuracy for the 'selected' subgroup by {avg_improvement_sel:.4f}.")
else:
    print(f"❌ On average, fine-tuning did not improve the accuracy for the 'selected' subgroup (Δ = {avg_improvement_sel:.4f}).")


