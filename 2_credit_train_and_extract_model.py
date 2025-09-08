import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from fairlearn.datasets import fetch_credit_card
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from numpy.polynomial.chebyshev import chebfit, Chebyshev, chebpts1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# For reproducibility
SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

appr_flag = 0 # 0: no approximation for nonlinear functions, 1: approximation

# --- Configuration ---
FINETUNE_SPLIT_FRAC = 0.90 # 95% of selected samples for fine-tuning, 5% remain in base training
h = [32, 64, 32, 16, 1]
ell = 4
b = 16 # The number of values in a batch
ROUND = 1
BASE_LR = 0.01
FT_LR = 0.05

def LSPA(fun, X, degree=3):
    # We determine the min and max values of all dps and features
    X = np.array([k for kk in X for k in kk])
    x_max = math.ceil(np.max(X))
    x_min = math.floor(np.min(X))
    n_samples = 100_000
    linspace = np.linspace(-1, 1, n_samples)
    x = np.sign(linspace) * np.abs(linspace)**3
    
    x = 0.5 * (x_max - x_min) * x + 0.5 * (x_max + x_min)

    y = fun(x)

    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    return poly

def chebyshev_approximation(fun, X, degree=3):
    a = math.floor(np.min(X))
    b = math.ceil(np.max(X))

    x_cheb = chebpts1(10_000)
    x = 0.5 * (b - a) * x_cheb + 0.5 * (b + a)
    y = fun(x)
    
    coeffs = chebfit(x, y, degree, domain=[a, b])
    cheb = Chebyshev(coeffs, domain=[a, b])
    return coeffs

def sigmoid(x):
    """Compute the derivative of the softplus function (which is the sigmoid function)."""
    return 1 / (1 + np.exp(-x)) 

def softplus(x):
    return np.log1p(np.exp(x)) 

def derivative_sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    df = f * (1 - f)
    return df

def Replicate(v, k, gap):
    gap = int(gap)
    v_extended = v + [0] * gap
    v_replicated = v_extended * k
    return v_replicated

def Flatten(W, gap, dim):
    if dim == 'r':
        num_rows = len(W)
        num_cols = len(W[0])
        flattened_W = []
        for row in range(num_rows):
            for col in range(num_cols):
                flattened_W.append(W[row][col])
            flattened_W += [0] * gap
    elif dim == 'c':
        num_rows = len(W)
        num_cols = len(W[0])
        flattened_W = []
        for col in range(num_cols):
            for row in range(num_rows):
                flattened_W.append(W[row][col])
            flattened_W += [0] * gap
    elif dim == '.':
        new_shape = len(W) * (gap + 1)
        flattened_W = [0] * new_shape
        flattened_W[::gap + 1] = W
    return flattened_W


def AlternatingPacking(X, y, X_t, W, h, ell):
    # d = h[0]
    packed_X = [None for _ in range(len(X))]
    packed_y = [None for _ in range(len(y))]
    packed_X_t = [None for _ in range(len(X_t))]
    packed_W = [None for _ in range(ell)]

    ft_num = len(X[0])
    padding = 0
    while pow(2, int(math.log2(ft_num))) != ft_num:
        ft_num += 1
        padding += 1
    
    # print("Feature number is padded to", ft_num, "from", len(X[0]), "with padding of", padding)

    gap = max(h[2] - h[0], 0)
    for n in range(len(X)):
        while len(X[n]) != ft_num:
            X[n].append(0)
        packed_X[n] = Replicate(X[n], h[1], gap)
    for n in range(len(X_t)):
        while len(X_t[n]) != ft_num:
            X_t[n].append(0)
            
        packed_X_t[n] = Replicate(X_t[n], h[1], gap)
    
    if ell%2==1:
        gap_y = h[ell]
        for n in range(len(X)):
            packed_y[n] = Flatten(y[n], gap_y, '.')
            packed_y[n] += [0] * (h[1] * h[2] - h[4]) # !
    else:
        for n in range(len(X)):
            packed_y[n] = y[n]
            if type(packed_y[n]) is not list:
                packed_y[n] = [packed_y[n]]
            packed_y[n] += [0] * (h[1] * h[2] - h[4]) # The first two elements are full so we have to pad the rest
    
    for j in range(ell):
        if j % 2 == 1: 
            gap = 0
            if h[j-1] > h[j+1]:
                gap = h[j-1] - h[j+1]
            print(h[j-1], ">?", h[j+1] , "Packing layer", j, "with gap", gap)
            packed_W[j] = Flatten(W[j], gap, 'r')
        else:
            gap = 0
            if h[j+2] > h[j]:
                gap = h[j+2] - h[j]
            print(h[j+2], ">?", h[j] , "Packing layer", j, "with gap", gap)
            packed_W[j] = Flatten(W[j], gap, 'c')

    return packed_X, packed_X_t, packed_y, packed_W


def RIS(c, p, s):
    for i in range(int(math.log2(s))):
        c += np.roll(c, -p*(pow(2, i)))
    return c

def RR(c, p, s):
    for i in range(int(math.log2(s))):
        c += np.roll(c, p*(pow(2, i)))
    return c

def save_list_to_txt(filename, data):
    with open(filename, "w") as f:
        for item in data:
            if isinstance(item, list):
                f.write(' '.join(map(str, item)) + '\n')  # multi-dimensional
            else:
                f.write(str(item) + '\n')  # 1D list


def build_model(input_shape, lr):
    """Builds and compiles a new Keras model."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='softplus', use_bias=False),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)),
        tf.keras.layers.Dense(32, activation='softplus', use_bias=False),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED+1)),
        tf.keras.layers.Dense(16, activation='softplus', use_bias=False),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED+2)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False),#, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED+3)),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":

    # Load and prepare data
    X, y = fetch_credit_card(return_X_y=True, as_frame=True)
    y = y.map({'0': 0, '1': 1})

    # Zero-pad all samples so that the number of features is equal to its next power of 2 #!
    num_features = len(X.columns)
    padding = 0
    while pow(2, int(math.log2(num_features))) != num_features:
        num_features += 1
        padding += 1
    # print("Feature number is padded to", num_features, "from", len(X.columns), "with padding of", padding)
    old_index = X.index  # save before converting
    X = X.to_numpy().tolist()  # Convert to list of lists
    for i in range(len(X)):
        while len(X[i]) < num_features:
            X[i].append(0)  # Zero-pad to the next power of 2
    X = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(num_features)], index=old_index)

    # Create Train/Validation/Test splits (70% / 15% / 15%)
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, train_size=0.7, random_state=SEED, stratify=y
    )

    X_val, X_test, y_val, y_test_o = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=SEED, stratify=y_test_val 
    )

    # Feature selection using only training set #!
    # rf = RandomForestClassifier(random_state=SEED, n_estimators=200)
    # rf.fit(X_train, y_train)

    # importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    # top16_features = importances.sort_values(ascending=False).head(16).index.tolist()

    # print("Top 16 features (selected on training):", top16_features)

    # Apply the same feature selection everywhere
    # X = X[top16_features]
    # X_train = X_train[top16_features]
    # X_val   = X_val[top16_features]
    # X_test  = X_test[top16_features]

    # Prepare specialized datasets for fine-tuning from the TRAINING set
    # Isolate all selected school samples from the training set
    selected_indices_train = (
        (X_train['x6'] > 1.5) | 
        (X_train['x5'] < 25.5) 
    )
    
    X_train_edu_all = X_train[selected_indices_train]
    y_train_edu_all = y_train[selected_indices_train]

    X_train_finetune, X_train_remain, y_train_finetune, y_train_remain = train_test_split(
        X_train_edu_all, y_train_edu_all, train_size=FINETUNE_SPLIT_FRAC, random_state=SEED
    )
    X_train_base = pd.concat([X_train[~selected_indices_train], X_train_remain], ignore_index=True)
    y_train_base = pd.concat([y_train[~selected_indices_train], y_train_remain], ignore_index=True)


    print("Length of the base set:", len(X_train_base))
    print("Length of the finetuning set:", len(X_train_finetune))

    X = X.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test_o = y_test_o.reset_index(drop=True)

    # Scale numeric features
    scaler = StandardScaler()
    X_val_scaled_o = scaler.fit_transform(X_val)
    X_scaled = scaler.transform(X)

    X_train_finetune_scaled_o = scaler.transform(X_train_finetune)
    X_train_base_scaled_o = scaler.transform(X_train_base)
    X_test_scaled_o = scaler.transform(X_test)

    # Build and train the base model
    print("\n--- Training Base Model ---")
    base_model = build_model(X_train_base_scaled_o.shape[1], BASE_LR)
    base_model.fit(
        X_train_base_scaled_o, y_train_base,
        validation_data=(X_val_scaled_o, y_val),
        epochs=10,
        batch_size=16,
        verbose=0
    )

    # Evaluate the BASE model on the TEST set
    print("--- Evaluating Base Model on Test Set ---")
    y_pred_base = (base_model.predict(X_test_scaled_o).ravel() > 0.5).astype(int)
    acc_base_all = accuracy_score(y_test_o, y_pred_base)
    selected_indices_test = (
        (X_test['x6'] > 1.5) | 
        (X_test['x5'] < 25.5) 
    )

    y_pred_base_sel = y_pred_base[selected_indices_test]
    acc_base_sel = accuracy_score(y_test_o[selected_indices_test], y_pred_base_sel)

    print(f"Base Accuracy (All): {acc_base_all:.4f} | Base Accuracy (Selected): {acc_base_sel:.4f}")

    # Output model and data

    base_weights = base_model.get_weights()
    base_weights_list = []

    for i, layer in enumerate(base_weights):
        # print(f"Layer {i} shape: {layer.shape}")
        as_list = layer.tolist()
        base_weights_list.append(as_list)


    # Pack the weights, X, and y
    X_train_finetune_scaled = X_train_finetune_scaled_o.tolist()
    X_test_scaled = X_test_scaled_o.tolist()
    y_train_finetune = y_train_finetune.tolist()
    y_test = y_test_o.tolist()

    limit_finetune = 2000 #! Set the size of the finetuning set
    X_train_finetune_scaled = X_train_finetune_scaled[:limit_finetune]
    y_train_finetune = y_train_finetune[:limit_finetune]
    print(f"Size of the finetuning set: {len(X_train_finetune_scaled)}")
    packed_X, packed_X_t, packed_y, packed_W = AlternatingPacking(X_train_finetune_scaled, y_train_finetune, X_test_scaled, base_weights_list, h, ell)

    # print(f"Packed X shape: {len(packed_X)} samples, each with {len(packed_X[0])} features")

    # ! Saving the model and data
    # save_list_to_txt("./txts/packed_X.txt", packed_X)
    # save_list_to_txt("./txts/packed_X_t.txt", packed_X_t)
    # save_list_to_txt("./txts/packed_y.txt", packed_y)
    # save_list_to_txt("./txts/packed_W.txt", packed_W)
    # save_list_to_txt("./txts/y_test.txt", y_test)

    appr_sigmoid = LSPA(sigmoid, X_scaled)
    appr_softplus = LSPA(softplus, X_scaled) 
    print("Sigmoid coeffs:", appr_sigmoid.coeffs)
    print("Softplus coeffs:", appr_softplus.coeffs)

    # Generating the masks 
    m1 = [0] * (h[1] * h[2])
    for i in range(len(m1)):
        if i % h[2] == 0:
            m1[i] = 1

    m2 = [0] * (h[1] * h[2])
    for i in range(len(m2)):
        if i < h[2]:
            m2[i] = 1

    m3 = [0] * (h[2] * h[3])
    for i in range(len(m3)):
        if i % h[2] == 0:
            m3[i] = 1

    m4 = [0] * (h[2] * h[3])
    for i in range(len(m4)):
        if i < h[4]:
            m4[i] = 1

    # print(packed_W[3])

    # Training starts
    m = math.ceil(len(packed_X) / b) # The number of batches
    print(f"Number of batches: {m}, batch size: {b}")
    all_vals = []
    for round in range(ROUND):
        for t in range(m): # later to be converted to batch
            d_W_4 = 0
            flag = False
            b_upt = b
            for item in range(b): # This portion needs to be parallelized
                if t*b+item < len(packed_X):
                    ### Forward Pass
                    L_0 = packed_X[t*b+item]

                    # Layer 1: 64 nodes
                    U_1 = np.multiply(L_0, packed_W[0])
                    U_1 = RIS(U_1, 1, h[0]) 
                    U_1 = np.multiply(U_1, m1)
                    U_1 = RR(U_1, 1, h[2])

                    # all_vals.append(max(U_1))
                    # all_vals.append(min(U_1))
                    if appr_flag == 1:
                        L_1 = appr_softplus(U_1)
                    else:
                        L_1 = softplus(U_1)

                    
                    # Layer 2: 32 nodes
                    U_2 = np.multiply(L_1, packed_W[1])
                    U_2 = RIS(U_2, h[2], h[1]) # h[0], h[1]
                    U_2 = np.multiply(U_2, m2)
                    U_2 = RR(U_2, h[2], h[3])

                    # all_vals.append(max(U_2))
                    # all_vals.append(min(U_2))
                    if appr_flag == 1:
                        L_2 = appr_softplus(U_2)
                    else:
                        L_2 = softplus(U_2)

                    L_2 = L_2[:h[2]*h[3]]  # Ensure L_2 has the correct shape for the next layer

                    # Layer 3: 16 nodes
                    U_3 = np.multiply(L_2, packed_W[2])
                    U_3 = RIS(U_3, 1, h[2])
                    U_3 = np.multiply(U_3, m3)
                    U_3 = RR(U_3, 1, h[4])

                    # all_vals.append(max(U_3))
                    # all_vals.append(min(U_3))
                    if appr_flag == 1:
                        L_3 = appr_softplus(U_3)
                    else:
                        L_3 = softplus(U_3)
    
                    # Layer 4: 1 node
                    U_4 = np.multiply(L_3, packed_W[3])
                    U_4 = RIS(U_4, h[2], h[3])
                    U_4 = np.multiply(U_4, m4)

                    all_vals.append(max(U_4))
                    all_vals.append(min(U_4))
                    if appr_flag == 1:
                        L_4 = appr_sigmoid(U_4)
                    else:
                        L_4 = sigmoid(U_4)

                    ### Backpropagation
                    E_4 = L_4 - packed_y[t*b+item][:h[2]*h[3]]

                    E_4 = np.multiply(E_4, m4)
                    E_4 = RR(E_4, h[2], h[3])       

                    d_W_4 += np.multiply(L_3, E_4)
                    # print(d_W_4[0])
                 
                else:
                    if flag == False:
                        b_upt = item
                    flag = True

            d_W_4_scaled = np.multiply(d_W_4, FT_LR/b_upt) 
            packed_W[3] -= d_W_4_scaled
            print(packed_W[3][0])
    
    # print(max(all_vals))
    # print(min(all_vals))
    # print(packed_W[3])


    # Evaluation 
    corr = 0
    corr_selected = 0
    for t in range(len(packed_X_t)): # later to be converted to batch
        ### Forward Pass
        L_0 = packed_X_t[t]

        # Layer 1: 64 nodes
        U_1 = np.multiply(L_0, packed_W[0])
        U_1 = RIS(U_1, 1, h[0]) 
        U_1 = np.multiply(U_1, m1)
        U_1 = RR(U_1, 1, h[2])
        L_1 = softplus(U_1)

        # Layer 2: 32 nodes
        U_2 = np.multiply(L_1, packed_W[1])
        U_2 = RIS(U_2, h[2], h[1]) # h[0], h[1]
        U_2 = np.multiply(U_2, m2)
        U_2 = RR(U_2, h[2], h[3])
        L_2 = softplus(U_2)

        L_2 = L_2[:h[2]*h[3]]  # Ensure L_2 has the correct shape for the next layer

        # Layer 3: 16 nodes
        U_3 = np.multiply(L_2, packed_W[2])
        U_3 = RIS(U_3, 1, h[2])
        U_3 = np.multiply(U_3, m3)
        U_3 = RR(U_3, 1, h[4])
        L_3 = softplus(U_3)

        # Layer 4: 1 node
        U_4 = np.multiply(L_3, packed_W[3])
        U_4 = RIS(U_4, h[2], h[3])
        U_4 = np.multiply(U_4, m4)
        L_4 = sigmoid(U_4)

        if L_4[0] > 0.5:
            y = 1
        else:
            y = 0

        if y_test[t] == 1 and y == 1:
            corr += 1
        elif  y_test[t] == 0 and y == 0:
            corr += 1

        if selected_indices_test.iloc[t]:
            if L_4[0] > 0.5:
                y = 1
            else:
                y = 0

            if y_test[t] == 1 and y == 1:
                corr_selected += 1
            elif  y_test[t] == 0 and y == 0:
                corr_selected += 1

    print(f"Test accuracy (All): {corr} {len(X_test_scaled)} %{corr*100/len(X_test_scaled):.4f} | Test accuracy (Selected): {corr_selected} {len(X_test_scaled_o[selected_indices_test])} %{corr_selected*100/len(X_test_scaled_o[selected_indices_test]):.4f}")
    chance_acc_all = max(y_test_o.value_counts()) / len(y_test_o)
    chance_acc_selected = max(y_test_o[selected_indices_test].value_counts()) / len(y_test_o[selected_indices_test])

    positive_indices = X_test.index[selected_indices_test]
    np.savetxt("./txts/chosen_indices.txt", positive_indices, fmt="%d")

    print(f"Chance-level Accuracy (All): {chance_acc_all:.4f}")
    print(f"Chance-level Accuracy (Selected): {chance_acc_selected:.4f}")