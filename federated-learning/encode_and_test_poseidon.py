import math
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.initializers import GlorotNormal
import sys

N = 10
h = [16, 64, 1]
items_per_client = math.floor(546 / N)
b = 32 # The number of values in a batch
ROUND = 50

# This is the batched version of poseidon_BLAS.py
def LSPA(fun, X, degree=3):
    # we determine the min and max values of all dps and features
    X = np.array([k for kk in X for k in kk])
    # X = np.array([np.mean(kk) for kk in X])
    x = np.linspace(math.floor(np.min(X)), math.ceil(np.max(X)), 10_000_000)
    y = fun(x)

    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    return poly

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
    packed_X = [[None for _ in range(len(X[i]))] for i in range(N)]
    packed_y = [[None for _ in range(len(y[i]))] for i in range(N)]
    packed_X_t = [None for _ in range(len(X_t))]
    packed_W = [None for _ in range(ell)]

    ft_num = len(X[0][0])
    padding = 0
    while pow(2, int(math.log2(ft_num))) != ft_num:
        ft_num += 1
        padding += 1
    
    for i in range(N):
        gap = max(h[2] - h[0], 0)
        for n in range(len(X[i])):
            while len(X[i][n]) != ft_num:
                X[i][n].append(0)
            packed_X[i][n] = Replicate(X[i][n], h[1], gap)
            # encode Xi
        for n in range(len(X_t)):
            while len(X_t[n]) != ft_num:
                X_t[n].append(0)
                
            packed_X_t[n] = Replicate(X_t[n], h[1], gap)
        
        if ell%2==1:
            gap_y = h[ell]
            for n in range(len(X[i])):
                packed_y[i][n] = Flatten(y[i][n], gap_y, '.')
                packed_y[i][n] += [0] * (h[0] * h[1] - h[2])
        else:
            for n in range(len(X[i])):
                packed_y[i][n] = y[i][n]
                if type(packed_y[i][n]) is not list:
                    packed_y[i][n] = [packed_y[i][n]]
                packed_y[i][n] += [0] * (h[0] * h[1] - h[2])
        #encode yi

        if i==0:
            for j in range(ell):
                if j % 2 == 1: 
                    if h[j-1] > h[j+1]:
                        gap = h[j-1] - h[j+1]
                    packed_W[j] = Flatten(W[j], gap, 'r')
                    # encrypt
                else:
                    if h[j+2] > h[j]:
                        gap = h[j+2] - h[j]
                    packed_W[j] = Flatten(W[j], gap, 'c')
                    # encrypt
    # print(packed_y)
    return packed_X, packed_X_t, packed_y, packed_W

def RIS(c, p, s):
    for i in range(int(math.log2(s))):
        c += np.roll(c, -p*(pow(2, i)))
    return c

def RR(c, p, s):
    for i in range(int(math.log2(s))):
        c += np.roll(c, p*(pow(2, i)))
    return c

def CreateModel(h):
    model = []
    initializer = GlorotNormal(seed=0)
    layer = initializer(shape=(9, h[1])).numpy().tolist()
    zero_part = np.zeros((7, h[1]))  # Generate (7, h[1]) zero values
    model.append(np.vstack((layer, zero_part)).tolist())
    model.append(initializer(shape=(h[1], h[2])).numpy().tolist())
    return model


def save_list_to_txt(filename, data):
    with open(filename, "w") as f:
        for item in data:
            if isinstance(item, list):
                f.write(' '.join(map(str, item)) + '\n')  # multi-dimensional
            else:
                f.write(str(item) + '\n')  # 1D list

def save_3d_list_to_txt(filename, data):
    """
    Save a 3D list to a file, separating 2D slices by blank lines.
    Each inner list is written as a space-separated line.
    """
    with open(filename, "w") as f:
        for i, matrix in enumerate(data):
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')
            if i < len(data) - 1:
                f.write('\n')  # Separate 2D matrices with a blank line
        f.write('\n')

if __name__ == "__main__":

    # Load dataset
    data_path = "./breast+cancer+wisconsin+original/breast-cancer-wisconsin.data"

    columns = ['ID', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 
            'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

    df = pd.read_csv(data_path, names=columns, na_values='?')

    # Check the first few rows of the dataset
    # print(df.head())

    # Drop rows with missing values
    df = df.dropna()

    # Separate features (X) and target (y)
    X = df.drop(columns=['ID', 'Class'])
    y = df['Class']

    # Convert target labels to binary (4 = malignant, 2 = benign)
    y = y.map({4: 1, 2: 0})
    # Convert labels to one-hot encoding #!
    # y = to_categorical(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features 
    scaler = StandardScaler() # assumes means and stds are shared amongst clients for standard scalar
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("StandardScaler done.")

    # Convert numpy ndarrays to lists
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    save_list_to_txt("./txts_dist/y_test.txt", y_test)

    appr_sigmoid = LSPA(sigmoid, X_train)
    appr_softplus = LSPA(softplus, X_train) 
    appr_der_sigmoid = LSPA(derivative_sigmoid, X_train)
    appr_der_sigmoid = np.polyder(appr_sigmoid)
    print(appr_sigmoid)
    print(appr_softplus)
    print(appr_der_sigmoid)

    # Take subset of x_train and y_train
    items_per_client = math.floor(len(X_train) / N)
    X_train_subsets = [None for _ in range(N)]
    y_train_subsets = [None for _ in range(N)]

    for i in range(N):
        X_train_subsets[i] = X_train[i*items_per_client:(i+1)*items_per_client]
        y_train_subsets[i] = y_train[i*items_per_client:(i+1)*items_per_client]
    
    ell = 2

    # Initialize the model weights
    model = CreateModel(h)

    # Pack the weights, X, and y
    packed_X, packed_X_t, packed_y, packed_W = AlternatingPacking(X_train_subsets, y_train_subsets, X_test, model, h, ell)

    save_3d_list_to_txt("./txts_dist/packed_X.txt", packed_X)
    save_list_to_txt("./txts_dist/packed_X_t.txt", packed_X_t)
    save_3d_list_to_txt("./txts_dist/packed_y.txt", packed_y)
    save_list_to_txt("./txts_dist/packed_W.txt", packed_W)

    # Generating the masks (each client should do this)
    m1 = [0] * (h[0] * h[1])
    for i in range(len(m1)):
        if i % h[0] == 0:
            m1[i] = 1

    m2 = [0] * (h[0] * h[1])
    for i in range(len(m2)):
        if i < h[2]:
            m2[i] = 1

    # Training starts
    m = math.ceil(items_per_client / b) # The number of batches

    for round in range(ROUND):
        print(round)
        for t in range(m): # later to be converted to batch
            d_W_1 = [0 for _ in range(N)]
            d_W_2 = [0 for _ in range(N)]
            for client_id in range(N):
                flag = False
                b_upt = b
                for item in range(b): # This portion needs to be parallelized
                    if t*b+item < len(packed_X[client_id]):
                        ### Forward Pass
                        L_0 = packed_X[client_id][t*b+item]
                        U_1 = np.multiply(L_0, packed_W[0])
                        U_1 = RIS(U_1, 1, h[0])
                        # print(U_1[:10])
                        U_1 = np.multiply(U_1, m1)
                        U_1 = RR(U_1, 1, h[2])
                        # Apply SmoothRELU function over the ndarray U_1 
                        L_1 = appr_softplus(U_1)   
                                     

                        U_2 = np.multiply(L_1, packed_W[1])
                        U_2 = RIS(U_2, h[0], h[1])
                        U_2 = np.multiply(U_2, m2)
                        # Apply Sigmoid function over the ndarray U_2
                        L_2 = appr_sigmoid(U_2)
                    
                        ### Backpropagation
                        # E_2 = 2 * (packed_y[client_id][t*b+item] - L_2)
                        E_2 = np.subtract(L_2, packed_y[client_id][t*b+item])
           
                        # Derivative of softplus: sigmoid
                        # d = derivative_sigmoid(U_2)
                        # d = np.multiply(d, m2) 
                        # E_2 = np.multiply(E_2, d)

                        E_2 = np.multiply(E_2, m2)
                        E_2 = RR(E_2, h[0], h[1])

                        d_W_2[client_id] += np.multiply(L_1, E_2)
                        E_1 = np.multiply(E_2, packed_W[1])
                        E_1 = RIS(E_1, 1, h[2])

                        # Derivative of softplus: sigmoid
                        d = appr_sigmoid(U_1)
                        # d = np.multiply(d, m1)
                        E_1 = np.multiply(E_1, d)

                        E_1 = np.multiply(E_1, m1)
                        E_1 = RR(E_1, 1, h[0])
                        d_W_1[client_id] += np.multiply(L_0, E_1)
           
                        # if client_id == 1:
                        #     print(" > ", np.multiply(L_0, E_1)[:10])
                    else:
                        if flag == False:
                            b_upt = item
                        flag = True
                # print(client_id, d_W_1[client_id][0])
            # print(d_W_1[0])
            eta = 0.1
            d_W_1_scaled = 0
            d_W_2_scaled = 0
            for j in range(N):
                d_W_1_scaled += np.multiply(d_W_1[j], eta/(b_upt*N))
                d_W_2_scaled += np.multiply(d_W_2[j], eta/(b_upt*N))
            # print(d_W_1_scaled[:10])
            packed_W[0] -= d_W_1_scaled
            packed_W[1] -= d_W_2_scaled
            print(packed_W[0][:5])
            # corr = 0
            # print(packed_X[0][0].tolist())corr = 0
            # for tt in range(len(packed_X_t)): 
            #     ### Forward Pass
            #     L_0 = packed_X_t[tt]

            #     U_1 = np.multiply(L_0, packed_W[0])
            #     U_1 = RIS(U_1, 1, h[0])
            #     U_1 = np.multiply(U_1, m1)
            #     U_1 = RR(U_1, 1, h[2])
            #     L_1 = appr_softplus(U_1)

            #     U_2 = np.multiply(L_1, packed_W[1])
            #     U_2 = RIS(U_2, h[0], h[1])
            #     U_2 = np.multiply(U_2, m2)
            #     L_2 = appr_sigmoid(U_2)
            #     # print(L_2[0], L_2[1])
            #     if L_2[0] >= L_2[1]:
            #         y = 0
            #     else:
            #         y = 1

            #     if y_test[tt] == [1, 0] and y == 0:
            #         corr += 1
            #     elif  y_test[tt] == [0, 1] and y == 1:
            #         corr += 1
            # # print with .4f
            # print(f"Round: {round}, Batch: {t}, %{corr*100/len(X_test):.4f}")

        


# Evaluation 
corr = 0
for t in range(len(packed_X_t)): 
    ### Forward Pass
    L_0 = packed_X_t[t]

    U_1 = np.multiply(L_0, packed_W[0])
    U_1 = RIS(U_1, 1, h[0])
    U_1 = np.multiply(U_1, m1)
    U_1 = RR(U_1, 1, h[2])
    L_1 = appr_softplus(U_1)

    U_2 = np.multiply(L_1, packed_W[1])
    U_2 = RIS(U_2, h[0], h[1])
    U_2 = np.multiply(U_2, m2)
    L_2 = appr_sigmoid(U_2)
    # print(L_2[0], L_2[1])
    if L_2[0] >= 0.5:
        y = 1
    else:
        y = 0

    if y_test[t] == 1 and y == 1:
        corr += 1
    elif  y_test[t] == 0 and y == 0:
        corr += 1

print(f"{corr} {len(X_test)} %{corr*100/len(X_test):.4f}")
    
