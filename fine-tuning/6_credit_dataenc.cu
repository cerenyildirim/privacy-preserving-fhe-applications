#include "heongpu.cuh"
#include "../example_util.h"
#include <chrono>
#include <stdlib.h>
#include <signal.h>
#include <iomanip>

// --------------------------------------------------------------
// File:    credit_v2.cu
// Author:  Ceren Yıldırım
// Created: 2025-12-08
// --------------------------------------------------------------

// --- ANSI color codes ---
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define DIM     "\033[2m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define WHITE_BG "\033[47;30m"

// Set to 1 to enable encrypted inference (0 for plaintext inference)
#define INFER_ENC 0

void printProgress(double progress) {
    int barWidth = 50;
    std::cout << CYAN << "  [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r" << RESET;
    std::cout.flush();
}

void sig_handler(int sig) { 
    std::cerr << RED << "\n[!] Signal ("<< sig <<") interrupted normal execution" << RESET << std::endl;
    abort();
    exit(EXIT_FAILURE);
}

struct EvalResults {
    double accuracy_full;     
    double accuracy_subset;   
    int correct_full;         
    int correct_subset;       
    int total_full;           
    int total_subset;         
};

void printCiphertext(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEDecryptor<heongpu::Scheme::CKKS>& decryptor, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder) {
    heongpu::Plaintext<heongpu::Scheme::CKKS> X_b(context);
    std::vector<double> X_b_vec;
    decryptor.decrypt(X_b, C);
    encoder.decode(X_b_vec, X_b);
    display_vector(X_b_vec, X_b_vec.size()/2, 5);
}

std::vector<std::vector<double>> read2DArrayTrain(const std::string& filename, int slot_count) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> current2D;
    std::string line;
    int slot_index = 0;
    std::vector<double> row;
    int count_rows = 0;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
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

std::vector<std::vector<double> > read2DArrayTest(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double> > data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;
        while (ss >> value) { 
            row.push_back(value);
        }
        data.push_back(row);
    }
    return data;
}

std::vector<std::vector<double> > read2DArrayModel(const std::string& filename, int slot_count, std::vector<int> h) {
    std::ifstream file(filename);
    std::vector<std::vector<double> > data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        for (int i = 0; i < (slot_count / (h[1]*h[2])) ; i++) {
            std::stringstream ss(line);
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

std::vector<int> readChosenIndices(const std::string& filename) {
    std::ifstream file(filename);
    int idx;
    std::vector<int> indices;
    while (file >> idx) {
        indices.push_back(idx);
    }
    return indices;
}

void RIS(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, int p, int s, heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> copyC(context);
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        operators.rotate_rows(C, copyC, galois_key, shift); 
        operators.add_inplace(C, copyC);
    }
}

void RR(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, int p, int s, heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> copyC(context);
    for (int i = 0; i < log2(s); i++) {
        int shift = - p * pow(2, i);
        operators.rotate_rows(C, copyC, galois_key, shift); 
        operators.add_inplace(C, copyC);
    }
}

void Softplus(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS>& I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS> &relin_key, double scale, double slot_count, heongpu::HEDecryptor<heongpu::Scheme::CKKS>& decryptor) {   
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I2(context);
    operators.multiply(I, I, I2); 
    operators.relinearize_inplace(I2, relin_key);
    operators.rescale_inplace(I2);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> a3I(context);
    operators.multiply_plain(I, PA3, a3I, scale); 
    operators.rescale_inplace(a3I);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I3(context);
    operators.multiply(I2, a3I, I3); 
    operators.relinearize_inplace(I3, relin_key);
    operators.rescale_inplace(I3);

    operators.multiply_plain_inplace(I2, PA2, scale); 
    operators.rescale_inplace(I2);

    while (I2.depth() < I3.depth()) { operators.mod_drop_inplace(I2); }

    operators.add(I3, I2, R); 
    operators.multiply_plain_inplace(I, PA1, scale); 
    operators.rescale_inplace(I);

    while (I.depth() < R.depth()) { operators.mod_drop_inplace(I); }
    operators.add_inplace(R, I);
    operators.add_plain_inplace(R, PA0);
}

void Sigmoid(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS>& I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key, double scale, double slot_count) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I2(context);
    operators.multiply(I, I, I2);
    operators.relinearize_inplace(I2, relin_key);
    operators.rescale_inplace(I2);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> a3I(context);
    operators.multiply_plain(I, PA3, a3I, scale); 
    operators.rescale_inplace(a3I);
    
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I3(context);
    operators.multiply(I2, a3I, I3); 
    operators.relinearize_inplace(I3, relin_key);
    operators.rescale_inplace(I3);

    operators.multiply_plain_inplace(I2, PA2, scale); 
    operators.rescale_inplace(I2);

    while (I2.depth() < I3.depth()) { operators.mod_drop_inplace(I2); }
    operators.add(I3, I2, R); 

    operators.multiply_plain_inplace(I, PA1, scale); 
    operators.rescale_inplace(I);

    while (I.depth() < R.depth()) { operators.mod_drop_inplace(I); }
    operators.add_inplace(R, I);
    operators.add_plain_inplace(R, PA0); 
}

void PMultiply(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (int i = 0; i < A.size(); i++) { C[i] = A[i] * B[i]; }
}

void PAddition(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (int i = 0; i < A.size(); i++) { C[i] = A[i] + B[i]; }
}

void PSubtraction(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (int i = 0; i < A.size(); i++) { C[i] = A[i] - B[i]; }
}

void PRIS(std::vector<double>& C, int p, int s) {
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        std::vector<double> copyC = C;
        rotate(C.begin(), C.begin() + shift, C.end());
        PAddition(C, C, copyC);
    }
}

void PRR(std::vector<double>& C, int p, int s) {
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        std::vector<double> copyC = C;
        rotate(C.rbegin(), C.rbegin() + shift, C.rend());
        PAddition(C, C, copyC);
    }
}

std::vector<double> RSigmoid(std::vector<double>& C) {
    std::vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) { R[i] = 1.0 / (1.0 + std::exp(-C[i])); }
    return R;
}

std::vector<double> RSoftplus(std::vector<double>& C) {
    std::vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        if (C[i] > 20) R[i] = C[i];            
        if (C[i] < -20) R[i] = std::exp(C[i]);
        R[i] = std::log1p(std::exp(C[i]));  
    }
    return R;
}

EvalResults evaluate_model(
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>>& C_X_t, std::vector<std::vector<double>>& y_t,
    std::vector<heongpu::Plaintext<heongpu::Scheme::CKKS>>& P_W, heongpu::Ciphertext<heongpu::Scheme::CKKS>& C_W_3,
    std::vector<int>& chosen_indices, int slot_count, std::vector<int>& h,
    std::vector<double>& m1, std::vector<double>& m2, std::vector<double>& m3, std::vector<double>& m4,
    double len_X_test, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators,
    heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key, heongpu::HEContext<heongpu::Scheme::CKKS>& context,
    heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::HEDecryptor<heongpu::Scheme::CKKS>& decryptor,
    heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key_rot, double scale, double number_of_items_in_ctext,
    const std::string& title = "Evaluation Results") 
{
    int corr = 0, corr_selected = 0;
    std::cout << CYAN << BOLD << "\n=== " << title << " ===" << RESET << "\n";
    std::cout << "  [*] Evaluating " << C_X_t.size() << " encrypted test batches...\n";

    for (int tt = 0; tt < C_X_t.size(); tt++) {
        std::cout << "\r  [>] Inferring batch " << tt + 1 << " / " << C_X_t.size() << std::flush;

        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_0(context);
        L_0 = C_X_t[tt];
        
        // ---- Layer 1 ----
        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);
        operators.multiply_plain(L_0, P_W[0], U_1); 
        operators.rescale_inplace(U_1);
        RIS(U_1, 1, h[0], galois_key_rot, operators, context);
        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1;
        encoder.encode(P_m1, m1, scale);
        while (P_m1.depth() < U_1.depth()) { operators.mod_drop_inplace(P_m1); }
        operators.multiply_plain_inplace(U_1, P_m1);
        operators.rescale_inplace(U_1);
        RR(U_1, 1, h[2], galois_key_rot, operators, context);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);
        Softplus(L_1, U_1, -2.38253701e-05,  3.51409155e-03,  8.49230659e-01,  1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);

        // ---- Layer 2 ----
        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);
        while (L_1.depth() > P_W[1].depth()) { operators.mod_drop_inplace(P_W[1]); }
        operators.multiply_plain(L_1, P_W[1], U_2);
        operators.rescale_inplace(U_2);
        RIS(U_2, h[2], h[1], galois_key_rot, operators, context);
        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
        encoder.encode(P_m2, m2, scale);
        while (U_2.depth() > P_m2.depth()) { operators.mod_drop_inplace(P_m2); }
        operators.multiply_plain_inplace(U_2, P_m2); 
        operators.rescale_inplace(U_2);
        RR(U_2, h[2], h[3], galois_key_rot, operators, context);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
        Softplus(L_2, U_2, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);

        // ---- Layer 3 ----
        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_3(context);
        while (L_2.depth() > P_W[2].depth()) { operators.mod_drop_inplace(P_W[2]); }
        operators.multiply_plain(L_2, P_W[2], U_3);
        operators.rescale_inplace(U_3);
        RIS(U_3, 1, h[2], galois_key_rot, operators, context);
        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m3;
        encoder.encode(P_m3, m3, scale);
        while (U_3.depth() > P_m3.depth()) { operators.mod_drop_inplace(P_m3); }
        operators.multiply_plain_inplace(U_3, P_m3); 
        operators.rescale_inplace(U_3);
        RR(U_3, 1, h[4], galois_key_rot, operators, context);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_3(context);
        Softplus(L_3, U_3, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);

        // ---- Layer 4 ----
        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_4(context);
        while (L_3.depth() > C_W_3.depth()) { operators.mod_drop_inplace(C_W_3); }
        while (C_W_3.depth() > L_3.depth()) { operators.mod_drop_inplace(L_3); }
        operators.multiply(L_3, C_W_3, U_4);
        operators.relinearize_inplace(U_4, relin_key);
        operators.rescale_inplace(U_4);
        RIS(U_4, h[2], h[3], galois_key_rot, operators, context);
        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m4;
        encoder.encode(P_m4, m4, scale);
        while (U_4.depth() > P_m4.depth()) { operators.mod_drop_inplace(P_m4); }
        while (P_m4.depth() > U_4.depth()) { operators.mod_drop_inplace(U_4); }
        operators.multiply_plain_inplace(U_4, P_m4); 
        operators.rescale_inplace(U_4);
        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_4(context);
        Sigmoid(L_4, U_4, 5.60468547e-06, -8.37716501e-04,  3.67742962e-02,  5.37938314e-01, operators, context, encoder, relin_key, scale, slot_count);

        heongpu::Plaintext<heongpu::Scheme::CKKS> L_4_plain(context);
        decryptor.decrypt(L_4_plain, L_4);
        std::vector<double> L_4_vec;
        encoder.decode(L_4_vec, L_4_plain);

        int item_size = slot_count / number_of_items_in_ctext;
        for (int i = 0; i < number_of_items_in_ctext; i++){
            if (tt * number_of_items_in_ctext + i >= len_X_test) break;
            int y_pred = (L_4_vec[i*item_size] > 0.5) ? 1 : 0;
            int y_true = int(y_t[tt*number_of_items_in_ctext+i][0]);
            if (y_pred == y_true) {
                corr++;
                if (std::find(chosen_indices.begin(), chosen_indices.end(), tt*number_of_items_in_ctext+i) != chosen_indices.end())
                    corr_selected++;
            }
        }
    }
    std::cout << std::endl;

    int total_full = len_X_test;
    int total_subset = chosen_indices.size();
    double acc_full = 100.0 * corr / total_full;
    double acc_subset = 100.0 * corr_selected / total_subset;

    std::cout << "  " << GREEN << "Full Set Correct: " << corr << " / " << total_full << " (" << std::fixed << std::setprecision(2) << acc_full << "%)" << RESET << "\n";
    std::cout << "  " << YELLOW << "Subset Correct  : " << corr_selected << " / " << total_subset << " (" << std::fixed << std::setprecision(2) << acc_subset << "%)" << RESET << "\n";

    return {acc_full, acc_subset, corr, corr_selected, total_full, total_subset};
}

EvalResults evaluate_model_ptext(
    std::vector<std::vector<double>>& packed_X_t, std::vector<std::vector<double>>& y_t,
    std::vector<std::vector<double>>& packed_W, std::vector<int>& chosen_indices,
    int slot_count, std::vector<int>& h, std::vector<double>& m1, std::vector<double>& m2,
    std::vector<double>& m3, std::vector<double>& m4, double len_X_test, double number_of_items_in_ctext,
    const std::string& title = "Evaluation Results") 
{
    int corr = 0, corr_selected = 0;
    std::cout << CYAN << BOLD << "\n=== " << title << " (Plaintext) ===" << RESET << "\n";
    std::cout << "  [*] Evaluating " << packed_X_t.size() << " test batches...\n";

    for (int tt = 0; tt < packed_X_t.size(); tt++) {
        std::cout << "\r  [>] Inferring batch " << tt + 1 << " / " << packed_X_t.size() << std::flush;
        std::vector<double> L_0 = packed_X_t[tt];

        // ---- Layers 1 to 4 ----
        std::vector<double> U_1(slot_count); PMultiply(U_1, L_0, packed_W[0]); PRIS(U_1, 1, h[0]); PMultiply(U_1, U_1, m1); PRR(U_1, 1, h[2]); std::vector<double> L_1 = RSoftplus(U_1);
        std::vector<double> U_2(slot_count); PMultiply(U_2, L_1, packed_W[1]); PRIS(U_2, h[2], h[1]); PMultiply(U_2, U_2, m2); PRR(U_2, h[2], h[3]); std::vector<double> L_2 = RSoftplus(U_2);
        std::vector<double> U_3(slot_count); PMultiply(U_3, L_2, packed_W[2]); PRIS(U_3, 1, h[2]); PMultiply(U_3, U_3, m3); PRR(U_3, 1, h[4]); std::vector<double> L_3 = RSoftplus(U_3);
        std::vector<double> U_4(slot_count); PMultiply(U_4, L_3, packed_W[3]); PRIS(U_4, h[2], h[3]); PMultiply(U_4, U_4, m4); std::vector<double> L_4 = RSigmoid(U_4);

        int item_size = slot_count / number_of_items_in_ctext;
        for (int i = 0; i < number_of_items_in_ctext; i++){
            if (tt * number_of_items_in_ctext + i >= len_X_test) break;
            int y_pred = (L_4[i*item_size] > 0.5) ? 1 : 0;
            int y_true = int(y_t[tt*number_of_items_in_ctext+i][0]);
            if (y_pred == y_true) {
                corr++;
                if (std::find(chosen_indices.begin(), chosen_indices.end(), tt*number_of_items_in_ctext+i) != chosen_indices.end())
                    corr_selected++;
            }
        }
    }
    std::cout << std::endl;

    int total_full = len_X_test;
    int total_subset = chosen_indices.size();
    double acc_full = 100.0 * corr / total_full;
    double acc_subset = 100.0 * corr_selected / total_subset;

    std::cout << "  " << GREEN << "Full Set Correct: " << corr << " / " << total_full << " (" << std::fixed << std::setprecision(2) << acc_full << "%)" << RESET << "\n";
    std::cout << "  " << YELLOW << "Subset Correct  : " << corr_selected << " / " << total_subset << " (" << std::fixed << std::setprecision(2) << acc_subset << "%)" << RESET << "\n";

    return {acc_full, acc_subset, corr, corr_selected, total_full, total_subset};
}

std::string get_env(const char* key) {
    if (key == nullptr) throw std::invalid_argument("Null pointer passed as environment variable name");
    if (*key == '\0') throw std::invalid_argument("Value requested for the empty-name environment variable");
    const char* ev_val = getenv(key);
    if (ev_val == nullptr) throw std::runtime_error("Environment variable not defined");
    return std::string { ev_val };
}

int main(int argc, char* argv[])
{
    std::cout << WHITE << DIM << "PID: process " << ::getpid() << " (parent: " << ::getppid() << ")" << RESET << std::endl;
    
    try {
        int defaultcudaDevice = 0;

        std::cout << CYAN << BOLD << "\n===========================================================" << RESET << std::endl;
        std::cout << CYAN << BOLD << "       HEonGPU HOMOMORPHIC FINETUNING DEMO" << RESET << std::endl;
        std::cout << CYAN << BOLD << "===========================================================\n" << RESET << std::endl;

        double len_X_train = 2000; 
        double len_X_test = 4500; 
        double ell = 4; 
        double LR = 0.05; 
        double round = 1; 
        std::vector<int> h = {32, 64, 32, 16, 1}; 
        std::string data_dir = "/home/ceren.yildirim/HEonGPU/example/mine/txts";   
            
        std::unordered_map<std::string, std::string> args;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--") {
                size_t eq_pos = arg.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = arg.substr(2, eq_pos - 2);
                    std::string value = arg.substr(eq_pos + 1);
                    args[key] = value;
                }
            }
        }

        if (args.find("train_size") != args.end()) len_X_train = std::stoi(args["train_size"]);
        if (args.find("epochs") != args.end()) round = std::stoi(args["epochs"]);
        if (args.find("learning_rate") != args.end()) LR = std::stod(args["learning_rate"]);
        if (args.find("data_dir") != args.end()) data_dir = args["data_dir"];
        if (args.find("gpu_device") != args.end()) defaultcudaDevice = std::stoi(args["gpu_device"]);
        bool paths_envvar = (args.find("paths_envvar") != args.end());

        cudaSetDevice(defaultcudaDevice); 
        std::cout << "  Utilized CUDA Device: " << YELLOW << defaultcudaDevice << RESET << std::endl;

        std::cout << MAGENTA << BOLD << "\n[1/6] INITIALIZING CONTEXT & KEYS..." << RESET << std::endl;

        heongpu::HEContext<heongpu::Scheme::CKKS> context(
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::sec128); 

        size_t poly_modulus_degree = 65536;
        const int slot_count = poly_modulus_degree / 2;
        context.set_poly_modulus_degree(poly_modulus_degree);

        std::cout << "  CKKS Ring Dimension: " << YELLOW << poly_modulus_degree << RESET << std::endl;

        int number_of_moduli = 30;
        int scale_bits = 50;
        std::vector<int> moduli(number_of_moduli, scale_bits);
        moduli.insert(moduli.begin(), 60);
        context.set_coeff_modulus_bit_sizes(moduli, {60, 60, 60}); 
        context.generate();

        double scale = pow(2.0, scale_bits);

        cudaDeviceSynchronize();
        auto start_keygen = std::chrono::high_resolution_clock::now();
        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context, 16); 
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
        heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
        heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context, encoder);

        int StoC_piece = 3;
        heongpu::BootstrappingConfig boot_config(3, StoC_piece, 10, false);
        operators.generate_bootstrapping_params(scale, boot_config, heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);

        std::vector<int> key_index = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, -1, -2, -4, -8, -16, -32, -64, -128, -256};
        heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key_rot(context, 14);

        keygen.generate_galois_key(galois_key_rot, secret_key, heongpu::ExecutionOptions().set_storage_type(heongpu::storage_type::DEVICE));
        cudaDeviceSynchronize();
        auto end_keygen = std::chrono::high_resolution_clock::now();
        
        std::cout << GREEN << "  [+] Key Generation finished in " << std::chrono::duration<double>(end_keygen - start_keygen).count() << " seconds!" << RESET << "\n";

        std::cout << MAGENTA << BOLD << "\n[2/6] LOADING DATASETS & MODEL..." << RESET << std::endl;
        std::cout << "  [*] Training size: " << len_X_train << " | Epochs: " << round << " | LR: " << LR << "\n";

        double number_of_items_in_ctext = ceil(slot_count / (h[1] * h[2]));
        int model_repeat = slot_count / (h[1] * h[2]); 
        double b = number_of_items_in_ctext / model_repeat; 
        double m = ceil(len_X_train / number_of_items_in_ctext); 
        double m_test = ceil(len_X_test / number_of_items_in_ctext); 

        std::vector<std::vector<double>> packed_X(m, std::vector<double>(slot_count)); 
        std::vector<std::vector<double>> packed_X_t(m, std::vector<double>(slot_count)); 
        std::vector<std::vector<double>> packed_y(m, std::vector<double>(slot_count)); 
        std::vector<std::vector<double>> y_t(len_X_test, std::vector<double>(1)); 
        std::vector<std::vector<double>> packed_W(ell, std::vector<double>(slot_count)); 
        std::vector<int> chosen_indices;

        if (paths_envvar) std::cout << "  [!] paths_envvar provided. Using ENV variables." << std::endl; 
        else std::cout << "  [!] Fetching data from: " << DIM << data_dir << RESET << std::endl;
        
        packed_X = read2DArrayTrain(paths_envvar ? get_env("PACKED_X") : data_dir + "/packed_X.txt", slot_count);
        packed_X_t = read2DArrayTrain(paths_envvar ? get_env("PACKED_X_T") : data_dir + "/packed_X_t.txt", slot_count);
        packed_y = read2DArrayTrain(paths_envvar ? get_env("PACKED_Y") : data_dir + "/packed_y.txt", slot_count);
        y_t = read2DArrayTest(paths_envvar ? get_env("Y_TEST") : data_dir + "/y_test.txt");
        packed_W = read2DArrayModel(paths_envvar ? get_env("PACKED_W") : data_dir + "/packed_W.txt", slot_count, h);
        chosen_indices = readChosenIndices(paths_envvar ? get_env("CHOSEN_INDICES") : data_dir + "/chosen_indices.txt");

        std::cout << GREEN << "  [+] Data structures loaded.\n" << RESET;

        std::vector<double> m1(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i % h[2] == 0) m1[i] = 1; }
        std::vector<double> m2(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i%(h[1] * h[2]) < h[2]) m2[i] = 1; }
        std::vector<double> m3(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i%(h[2]) == 0) m3[i] = 1; }
        std::vector<double> m4(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i%(h[2] * h[3]) < h[4] && i%(h[1] * h[2]) < h[2] * h[3]) m4[i] = 1; }

        std::vector<heongpu::Plaintext<heongpu::Scheme::CKKS>> P_W;
        for (int i = 0; i < ell; i++) {
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_W_i(context);
            encoder.encode(P_W_i, packed_W[i], scale);
            P_W.emplace_back(std::move(P_W_i));
        }

        heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_3(context);
        encryptor.encrypt(C_W_3, P_W[3]);
        std::cout << GREEN << "  [+] Plaintext masks and base model layer ciphertexts prepped.\n" << RESET;

        std::cout << MAGENTA << BOLD << "\n[3/6] ENCRYPTING DATA (CLIENT-SIDE)..." << RESET << std::endl;
        
        std::cout << "  [*] Encrypting Finetuning Samples (" << m << " batches)...\n";
        std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> C_X, C_y;
        cudaDeviceSynchronize();
        auto start_encryption = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < m; i++) {
            printProgress((double)(i + 1) / m);
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_X_i(context), P_y_i(context);
            encoder.encode(P_X_i, packed_X[i], scale);
            encoder.encode(P_y_i, packed_y[i], scale);

            heongpu::Ciphertext<heongpu::Scheme::CKKS> C_X_i(context), C_y_i(context);
            heongpu::ExecutionOptions options;
            options.set_storage_type(heongpu::storage_type::HOST).set_initial_location(true);
            
            encryptor.encrypt(C_X_i, P_X_i, options);
            encryptor.encrypt(C_y_i, P_y_i, options);
            
            C_X.emplace_back(std::move(C_X_i));
            C_y.emplace_back(std::move(C_y_i));
        }
        std::cout << std::endl;
        
        // Encrypt if INFER_ENC == 1
        std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> C_X_t;
        if (INFER_ENC == 1) {
            std::cout << "  [*] Encrypting Test Samples (" << m_test << " batches)...\n";
            for(int i = 0; i < m_test; i++) {
                printProgress((double)(i + 1) / m_test);
                heongpu::Plaintext<heongpu::Scheme::CKKS> P_X_t_i(context);
                encoder.encode(P_X_t_i, packed_X_t[i], scale);

                heongpu::Ciphertext<heongpu::Scheme::CKKS> C_X_t_i(context);
                heongpu::ExecutionOptions options;
                options.set_storage_type(heongpu::storage_type::HOST).set_initial_location(true);
                encryptor.encrypt(C_X_t_i, P_X_t_i, options);
                C_X_t.emplace_back(std::move(C_X_t_i));
            }
            std::cout << std::endl;
        }

        cudaDeviceSynchronize();
        auto end_encryption = std::chrono::high_resolution_clock::now();
        std::cout << GREEN << "  [+] Datasets encrypted in " << std::chrono::duration<double>(end_encryption - start_encryption).count() << " seconds!" << RESET << "\n";

        std::cout << MAGENTA << BOLD << "\n[4/6] PRE-EVALUATION (BEFORE FINETUNING)..." << RESET << std::endl;
        EvalResults before;
        if (INFER_ENC == 1) {
            before = evaluate_model(C_X_t, y_t, P_W, C_W_3, chosen_indices, slot_count, h, m1, m2, m3, m4, len_X_test, operators, relin_key, context, encoder, decryptor, galois_key_rot, scale, number_of_items_in_ctext, "Before Fine-tuning");
        } else {
            before = evaluate_model_ptext(packed_X_t, y_t, packed_W, chosen_indices, slot_count, h, m1, m2, m3, m4, len_X_test, number_of_items_in_ctext, "Before Fine-tuning");
        }

        std::cout << MAGENTA << BOLD << "\n[5/6] HOMOMORPHIC FINETUNING (SERVER-SIDE)..." << RESET << std::endl;
        double total_training_time = 0.0;

        for (int rn = 0; rn < round; rn++) {
            std::cout << BLUE << BOLD << "  [=== Epoch " << rn + 1 << " / " << int(round) << " ===]" << RESET << std::endl;

            for (int t = 0; t < m; t++) {
                cudaDeviceSynchronize();
                const auto start_batch = std::chrono::steady_clock::now();
            
                std::vector<double> all_zeros(slot_count, 0);
                heongpu::Plaintext<heongpu::Scheme::CKKS> all_zeros_plain(context);
                encoder.encode(all_zeros_plain, all_zeros, scale);
                heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_4(context);
                encryptor.encrypt(d_W_4, all_zeros_plain);

                std::cout << "\r    " << YELLOW << "Working ->" << RESET << " Batch " << t + 1 << " / " << m << std::flush;
                
                for (int item = 0; item < b; item++) {
                    if (t*b+item < len_X_train) {
                        heongpu::ExecutionOptions options;
                        options.set_storage_type(heongpu::storage_type::HOST).set_initial_location(true);
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_0(context, options);
                        L_0 = C_X[t*b+item];
                        L_0.store_in_device();

                        // ---- Layer 1 ----
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);
                        operators.multiply_plain(L_0, P_W[0], U_1); 
                        operators.rescale_inplace(U_1);
                        RIS(U_1, 1, h[0], galois_key_rot, operators, context);
                        
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1;
                        encoder.encode(P_m1, m1, scale);
                        while (P_m1.depth() < U_1.depth()) { operators.mod_drop_inplace(P_m1); }
                        operators.multiply_plain_inplace(U_1, P_m1);
                        operators.rescale_inplace(U_1);
                        RR(U_1, 1, h[2], galois_key_rot, operators, context);
                        
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);
                        Softplus(L_1, U_1, -2.38253701e-05,  3.51409155e-03,  8.49230659e-01,  1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);
                        
                        // ---- Layer 2 ----
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);
                        while (L_1.depth() > P_W[1].depth()) { operators.mod_drop_inplace(P_W[1]); }
                        operators.multiply_plain(L_1, P_W[1], U_2);
                        operators.rescale_inplace(U_2);
                        RIS(U_2, h[2], h[1], galois_key_rot, operators, context);
                    
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
                        encoder.encode(P_m2, m2, scale);
                        while (U_2.depth() > P_m2.depth()) { operators.mod_drop_inplace(P_m2); }
                        operators.multiply_plain_inplace(U_2, P_m2); 
                        operators.rescale_inplace(U_2);
                        RR(U_2, h[2], h[3], galois_key_rot, operators, context);

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
                        Softplus(L_2, U_2, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);

                        // ---- Layer 3 ----
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_3(context);
                        while (L_2.depth() > P_W[2].depth()) { operators.mod_drop_inplace(P_W[2]); }
                        operators.multiply_plain(L_2, P_W[2], U_3);
                        operators.rescale_inplace(U_3);
                        RIS(U_3, 1, h[2], galois_key_rot, operators, context);
                        
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m3;
                        encoder.encode(P_m3, m3, scale);
                        while (U_3.depth() > P_m3.depth()) { operators.mod_drop_inplace(P_m3); }
                        operators.multiply_plain_inplace(U_3, P_m3); 
                        operators.rescale_inplace(U_3);
                        RR(U_3, 1, h[4], galois_key_rot, operators, context);

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_3(context);
                        Softplus(L_3, U_3, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);

                        // ---- Layer 4 ----
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_4(context);
                        while (L_3.depth() > C_W_3.depth()) { operators.mod_drop_inplace(C_W_3); }
                        while (C_W_3.depth() > L_3.depth()) { operators.mod_drop_inplace(L_3); }
                        
                        operators.multiply(L_3, C_W_3, U_4);
                        operators.relinearize_inplace(U_4, relin_key);
                        operators.rescale_inplace(U_4);
                        RIS(U_4, h[2], h[3], galois_key_rot, operators, context);
                     
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m4;
                        encoder.encode(P_m4, m4, scale);
                        while (U_4.depth() > P_m4.depth()) { operators.mod_drop_inplace(P_m4); }
                        while (P_m4.depth() > U_4.depth()) { operators.mod_drop_inplace(U_4); }
                        
                        operators.multiply_plain_inplace(U_4, P_m4); 
                        operators.rescale_inplace(U_4);
                        
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_4(context);
                        Sigmoid(L_4, U_4, 5.60468547e-06, -8.37716501e-04,  3.67742962e-02,  5.37938314e-01, operators, context, encoder, relin_key, scale, slot_count);
                      
                        // ****************** Backpropagation ****************** //
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> E_4(context);
                        operators.negate(L_4, E_4); 
                        
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> C_y_curr(context, options);
                        C_y_curr = C_y[t*b+item];
                        C_y_curr.store_in_device();

                        while (C_y_curr.depth() < E_4.depth()) { operators.mod_drop_inplace(C_y_curr); }
                        operators.add_inplace(E_4, C_y_curr);
                        
                        while (E_4.depth() < P_m4.depth()) { operators.mod_drop_inplace(E_4); }
                        while (E_4.depth() > P_m4.depth()) { operators.mod_drop_inplace(P_m4); }

                        operators.multiply_plain_inplace(E_4, P_m4); 
                        operators.rescale_inplace(E_4);
                        RR(E_4, h[2], h[3], galois_key_rot, operators, context);

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> interm(context);
                        while (L_3.depth() < E_4.depth()) { operators.mod_drop_inplace(L_3); }
                        while (L_3.depth() > E_4.depth()) { operators.mod_drop_inplace(E_4); }

                        operators.multiply(E_4, L_3, interm); 
                        operators.relinearize_inplace(interm, relin_key);
                        operators.rescale_inplace(interm);
                        
                        while (d_W_4.depth() < interm.depth()) { operators.mod_drop_inplace(d_W_4); }
                        while (d_W_4.depth() > interm.depth()) { operators.mod_drop_inplace(interm); }
                        operators.add_inplace(d_W_4, interm);     
                    }
                }

                std::vector<double> eta(slot_count, LR/(b*number_of_items_in_ctext));
                heongpu::Plaintext<heongpu::Scheme::CKKS> eta_P(context); 
                encoder.encode(eta_P, eta, scale);

                heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_4_scaled;        
                encryptor.encrypt(d_W_4_scaled, all_zeros_plain);
                
                while (eta_P.depth() < d_W_4.depth()) { operators.mod_drop_inplace(eta_P); }
                operators.multiply_plain_inplace(d_W_4, eta_P);
                operators.rescale_inplace(d_W_4);

                while(d_W_4_scaled.depth() < d_W_4.depth()) { operators.mod_drop_inplace(d_W_4_scaled); }
                RIS(d_W_4, h[1]*h[2], model_repeat, galois_key_rot, operators, context);
                operators.add_inplace(d_W_4_scaled, d_W_4); 

                while (C_W_3.depth() < d_W_4_scaled.depth()) { operators.mod_drop_inplace(C_W_3); }
                operators.add_inplace(C_W_3, d_W_4_scaled);

                while (C_W_3.depth() < (number_of_moduli - StoC_piece)) { operators.mod_drop_inplace(C_W_3); }
                
                std::cout << "\r    " << CYAN << "Bootstrapping..." << RESET << std::string(10, ' ') << std::flush;
                heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_b3 = operators.slim_bootstrapping(C_W_3, galois_key_rot, relin_key);
                C_W_3 = C_W_b3;

                cudaDeviceSynchronize();
                const auto finish_batch = std::chrono::steady_clock::now();
                double batch_time = std::chrono::duration<double>(finish_batch - start_batch).count();
                total_training_time += batch_time;
                
                std::cout << "\r    " << GREEN << "-> Batch " << t + 1 << " completed in " << std::fixed << std::setprecision(2) << batch_time << " seconds." << RESET << std::string(15, ' ') << "\n";
            }
        }

        std::cout << GREEN << BOLD << "\n  [+] Finetuning finished in " << total_training_time << " seconds!" << RESET << std::endl;

        std::cout << MAGENTA << BOLD << "\n[6/6] POST-EVALUATION (AFTER FINETUNING)..." << RESET << std::endl;
        EvalResults after;
        if (INFER_ENC == 1) {
            after = evaluate_model(C_X_t, y_t, P_W, C_W_3, chosen_indices, slot_count, h, m1, m2, m3, m4, len_X_test, operators, relin_key, context, encoder, decryptor, galois_key_rot, scale, number_of_items_in_ctext, "After Fine-tuning");
        } else {
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_W_3(context);
            std::vector<double> W_3;
            decryptor.decrypt(P_W_3, C_W_3);
            encoder.decode(W_3, P_W_3);
            packed_W[3] = W_3;

            after = evaluate_model_ptext(packed_X_t, y_t, packed_W, chosen_indices, slot_count, h, m1, m2, m3, m4, len_X_test, number_of_items_in_ctext, "After Fine-tuning");
        }

        double improvement = after.accuracy_subset - before.accuracy_subset;
        std::cout << BLUE << BOLD << "\n===========================================================" << RESET << std::endl;
        std::cout << BLUE << BOLD << "  [!] SUBSET ACCURACY IMPROVEMENT: " << (improvement > 0 ? GREEN : RED) << BOLD << "+" << std::fixed << std::setprecision(2) << improvement << "%" << RESET << std::endl;
        std::cout << BLUE << BOLD << "===========================================================\n" << RESET << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << RED << "\nError occurred: " << ex.what() << RESET << std::endl;
        exit (EXIT_FAILURE);
    } catch (...) {
        std::cerr << RED << "\nAn exception occured and program is abruptly terminated" << RESET << std::endl;
        exit (EXIT_FAILURE);
    }    
    return EXIT_SUCCESS;
}
