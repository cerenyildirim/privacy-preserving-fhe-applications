#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>       // for log2, ceil, etc.
#include <algorithm>   // for rotate

using namespace std;


vector<vector<double>> read2DArrayTrain(const string& filename, int slot_count) {
    ifstream file(filename);
    vector<vector<double>> current2D;
    string line;
    int slot_index = 0;
    vector<double> row;
    int count_rows = 0;

    while (getline(file, line)) {
        
        istringstream ss(line);
        double value;
        while (ss >> value) {
            if(slot_index == slot_count) {
                current2D.push_back(row);
                row.clear();
                slot_index = 0;
                count_rows++;
            }
            row.push_back(value);
            slot_index++;
        }
    }

    while (slot_index < slot_count){
        row.push_back(0);
        slot_index++;
    }
    current2D.push_back(row);

    return current2D;
}

vector<vector<double> > read2DArrayTest(const string& filename) {
    ifstream file(filename);
    vector<vector<double> > data;
    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        double value;
        while (ss >> value) { 
            row.push_back(value);
        }
        data.push_back(row);
    }
    return data;
}

vector<vector<double> > read2DArrayModel(const string& filename, int slot_count, vector<int> h) {
    ifstream file(filename);
    vector<vector<double> > data;
    string line;
    
    while (getline(file, line)) {
        vector<double> row;
        for (int i = 0; i < (slot_count / (h[1]*h[2])) ; i++) {
            stringstream ss(line);
            double value;
            while (ss >> value) { 
                row.push_back(value);
            }
            while(row.size() % (h[1]*h[2]) != 0) {
                row.push_back(0);
            }
        }

        data.push_back(row);
    }
    return data;
}

vector<int> readChosenIndices(const string& filename) {
    std::ifstream file(filename);
    int idx;
    std::vector<int> indices;
    while (file >> idx) {
        indices.push_back(idx);
    }
    return indices;
}

void Multiply(vector <double>& C, vector<double>& A, vector<double>& B) {
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] * B[i];
    }
}

void Addition(vector <double>& C, vector<double>& A, vector<double>& B) {
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] + B[i];
    }
}

void Subtraction(vector <double>& C, vector<double>& A, vector<double>& B) {
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] - B[i];
    }
}

void RIS(vector<double>& C, int p, int s) {
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        vector<double> copyC = C;
        rotate(C.begin(), C.begin() + shift, C.end());
        Addition(C, C, copyC);
    }
}

void RR(vector<double>& C, int p, int s) {
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        vector<double> copyC = C;
        rotate(C.rbegin(), C.rbegin() + shift, C.rend());
        Addition(C, C, copyC);
    }
}

vector<double> Sigmoid(vector<double>& C) {
    vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        R[i] = 5.60468547e-06 * pow(C[i], 3) -8.37716501e-04 * pow(C[i], 2) + 3.67742962e-02* C[i] + 5.37938314e-01;
    }
    return R;
}

vector<double> Softplus(vector<double>& C) {
    vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        R[i] = -2.38253701e-05 * pow(C[i], 3) + 3.51409155e-03 * pow(C[i], 2) + 8.49230659e-01 * C[i] + 1.81535534e+00;
    }
    return R;
}

// Sigmoid function: 1 / (1 + exp(-x))
vector<double> RSigmoid(vector<double>& C) {
    vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        R[i] = 1.0 / (1.0 + std::exp(-C[i]));
    }
    return R;
}

// Softplus function: log(1 + exp(x))
vector<double> RSoftplus(vector<double>& C) {
    vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        // For numerical stability
        // if (C[i] > 20) 
        //     R[i] = C[i];             
        // if (C[i] < -20) 
        //     R[i] = std::exp(C[i]); 
        R[i] = std::log1p(std::exp(C[i])); 
    }
    return R;
}


int main() {

    double len_X_train = 2000; //4076; //!
    double len_X_test = 4500; //!
    double ell = 4;
    vector<int> h = {32, 64, 32, 16, 1}; //!

    double slot_count = 32768; // This is slot_count, not polymod degree!
    double number_of_items_in_ctext = ceil(slot_count / (h[1] * h[2]));

    std::cout << "The # of items in a ciphertext: " << number_of_items_in_ctext << std::endl;

    int model_repeat = slot_count / (h[1] * h[2]); // the number of times the model is repeated in a ciphertext, equal to number_of_items_in_ctext
    double b = number_of_items_in_ctext / model_repeat; // number of batch ciphertexts, ML batch is 8 and we can store model_repeat of the items in a single ciphertext
    double m = ceil(len_X_train / number_of_items_in_ctext); // the number of batches
    std::cout << "The # of batches (m): " << m << std::endl;
    std::cout << "The # of batch ciphertexts (b): " << b << std::endl;
    double round = 1;

    // read the encoded values
    vector<vector<double>> packed_X(m, vector<double>(slot_count));
    vector<vector<double>> packed_X_t(len_X_test, vector<double>(h[1]*h[2]));
    vector<vector<double>> packed_y(m, vector<double>(slot_count));
    vector<vector<double>> y_t(len_X_test, vector<double>(1)); 
    vector<vector<double>> packed_W(ell, vector<double>(slot_count));
    vector<int> chosen_indices;

    packed_X = read2DArrayTrain("../txts/packed_X.txt", slot_count);
    packed_X_t = read2DArrayTest("../txts/packed_X_t.txt");
    packed_y = read2DArrayTrain("../txts/packed_y.txt", slot_count);
    y_t = read2DArrayTest("../txts/y_test.txt");
    packed_W = read2DArrayModel("../txts/packed_W.txt", slot_count, h);
    chosen_indices = readChosenIndices("../txts/chosen_indices.txt");

    // generate masks
    vector<double> m1(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i % h[2] == 0)
            m1[i] = 1;
    }

    vector<double> m2(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[1] * h[2]) < h[2])
            m2[i] = 1;
    }

    vector<double> m3(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[2]) == 0)
            m3[i] = 1;
    }

    vector<double> m4(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[2] * h[3]) < h[4]) {
            if (i%(h[1] * h[2]) < h[2] * h[3])
                m4[i] = 1;
        }
    }


    // masks for the incomplete batch ciphertexts
    int full_slots = (int(len_X_train) % (int(slot_count) / int(h[1] * h[2])));
    vector<double> m1_i(slot_count, 0);
    std::cout << "Full slots: " << full_slots << std::endl;
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i % h[2] == 0)
            m1_i[i] = 1;
    }

    vector<double> m2_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i%(h[1] * h[2]) < h[2])
            m2_i[i] = 1;
    }

    vector<double> m3_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i%(h[2]) == 0)
            m3_i[i] = 1;
    }

    vector<double> m4_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i%(h[2] * h[3]) < h[4])
            m4_i[i] = 1;
    }
        
    std::cout << "Masks are generated" << std::endl;

    // training starts!
    for (int rn = 0; rn < round; rn++) {
        std::cout << "Round " << rn << std::endl;
        for (int t = 0; t < m; t++) {
            std::cout << "Batch " << t << std::endl;
            vector<double> d_W_4(slot_count, 0);
            for (int item = 0; item < b; item++) {
                if (t*b+item < len_X_train) { // fix this condition but it is negligible right now
                    
                    vector<double> L_0 = packed_X[t*b+item];

                    // Layer 1
                    vector<double> U_1(slot_count);
                    Multiply(U_1, L_0, packed_W[0]);
                    RIS(U_1, 1, h[0]);
                    Multiply(U_1, U_1, m1);
                    RR(U_1, 1, h[2]);
                    vector<double> L_1(slot_count);
                    L_1 = Softplus(U_1);

                    // Layer 2
                    vector<double> U_2(slot_count);
                    Multiply(U_2, L_1, packed_W[1]);
                    RIS(U_2, h[2], h[1]);
                    Multiply(U_2, U_2, m2);
                    RR(U_2, h[2], h[3]);
                    vector<double> L_2(slot_count);
                    L_2 = Softplus(U_2);

                    // Multiply(L_2, L_2, m_l1);

                    // Layer 3
                    vector<double> U_3(slot_count);
                    Multiply(U_3, L_2, packed_W[2]);
                    RIS(U_3, 1, h[2]);
                    Multiply(U_3, U_3, m3);
                    RR(U_3, 1, h[4]);
                    vector<double> L_3(slot_count);
                    L_3 = Softplus(U_3);

                    // Layer 4
                    vector<double> U_4(slot_count);
                    Multiply(U_4, L_3, packed_W[3]);
                    RIS(U_4, h[2], h[3]);
                    Multiply(U_4, U_4, m4);
                    vector<double> L_4(slot_count);
                    L_4 = Sigmoid(U_4);

                    // BACKPROPOGATION

                    vector<double> E_4(slot_count);
                    Subtraction(E_4, L_4, packed_y[t*b+item]);
                    vector<double> d(slot_count);
             
                    Multiply(E_4, E_4, m4);        
                    RR(E_4, h[2], h[3]);           
                    vector<double> interm(slot_count);
                    Multiply(interm, L_3, E_4);
                    Addition(d_W_4, d_W_4, interm);
        

                }
            }
        
            vector<double> eta(slot_count, 0.05/(b*number_of_items_in_ctext));
            vector<double> d_W_4_scaled(slot_count, 0);
            
            RIS(d_W_4, (h[1]*h[2]), int(slot_count / (h[1] * h[2])));
            Addition(d_W_4_scaled, d_W_4_scaled, d_W_4); 
        
            Multiply(d_W_4_scaled, d_W_4_scaled, eta);
            Subtraction(packed_W[3], packed_W[3], d_W_4_scaled);

            for (size_t i = 0; i < 1 ; ++i) {
                std::cout << packed_W[3][i] << " ";
            }
            // return 0;
        }
    }


    // write to file
    std::ofstream out_C_W_4("C_W_4_ptext.txt");
    if (out_C_W_4.is_open()) {
        for (const auto& val : packed_W[3]) {
            out_C_W_4 << val << std::endl;
        }
        out_C_W_4.close();      
        std::cout << "C_W_4 ciphertexts written to files C_W_4_ptext.txt" << std::endl;
    } else {
        std::cerr << "Unable to open files for writing C_W ciphertexts." << std::endl;
        return EXIT_FAILURE;
    }
    
    int corr = 0; int corr_selected = 0;
    for (int tt = 0; tt < packed_X_t.size(); tt++) {
        if (tt % 100 == 0) 
            std::cout << tt << "/" << packed_X_t.size() << std::endl;

        vector<double> L_0 = packed_X_t[tt];


        // Layer 1
        vector<double> U_1(slot_count);
        Multiply(U_1, L_0, packed_W[0]);
        RIS(U_1, 1, h[0]);
        Multiply(U_1, U_1, m1);
        RR(U_1, 1, h[2]);
        vector<double> L_1(slot_count);
        L_1 = RSoftplus(U_1);

        // Layer 2
        vector<double> U_2(slot_count);
        Multiply(U_2, L_1, packed_W[1]);
        RIS(U_2, h[2], h[1]);
        Multiply(U_2, U_2, m2);
        RR(U_2, h[2], h[3]);
        vector<double> L_2(slot_count);
        L_2 = RSoftplus(U_2);

        // Layer 3
        vector<double> U_3(slot_count);
        Multiply(U_3, L_2, packed_W[2]);
        RIS(U_3, 1, h[2]);
        Multiply(U_3, U_3, m3);
        RR(U_3, 1, h[4]);
        vector<double> L_3(slot_count);
        L_3 = RSoftplus(U_3);

        // Layer 4
        vector<double> U_4(slot_count);
        Multiply(U_4, L_3, packed_W[3]);
        RIS(U_4, h[2], h[3]);
        Multiply(U_4, U_4, m4);
        vector<double> L_4(slot_count);
        L_4 = RSigmoid(U_4);

        int y = 1;
        if (L_4[0] > 0.5)
            y = 1;
        else
            y = 0;

        if (int(y_t[tt][0]) == 1 && y == 1)
            corr += 1;
        else if  (int(y_t[tt][0]) == 0 && y == 0)
            corr += 1;

        if (std::find(chosen_indices.begin(), chosen_indices.end(), tt) != chosen_indices.end())
        {   
            if (L_4[0] > 0.5)
                y = 1;
            else
                y = 0;

            if (int(y_t[tt][0]) == 1 && y == 1)
                corr_selected += 1;
            else if  (int(y_t[tt][0]) == 0 && y == 0)
                corr_selected += 1;
        }
    }

    std::cout << corr << " " << len_X_test << " " << (corr / len_X_test) << std::endl;
    std::cout << corr_selected << " " << chosen_indices.size() << " " << double(corr_selected / double(chosen_indices.size())) << std::endl;
    
    return 0;
}