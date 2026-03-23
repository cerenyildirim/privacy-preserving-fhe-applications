#include "openfhe.h"
#include <chrono>
#include <stdlib.h>
#include <signal.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <iomanip>

using namespace lbcrypto;

// ==========================================
// ANSI Color Codes for Fancy Terminal Output
// ==========================================
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

void printCiphertext(Ciphertext<DCRTPoly>& C, CryptoContext<DCRTPoly>& context, PrivateKey<DCRTPoly>& secretKey) {
    Plaintext result;
    context->Decrypt(secretKey, C, &result);
    result->SetLength(C->GetSlots());
    std::vector<double> finalResult = result->GetRealPackedValue();

    std::cout << std::setprecision(5);
    std::cout << "[ ";
    for (size_t i = 0; i < finalResult.size(); ++i) {
        std::cout << finalResult[i] << " ";
    }
    std::cout << "... ]" << std::endl;
}

std::vector<std::vector<double>> read2DArrayTrain(const std::string& filename, int slot_count) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> current2D;
    std::string line;
    int slot_index = 0;
    std::vector<double> row;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        double value;
        while (ss >> value) {
            if(slot_index == slot_count) {
                current2D.push_back(row);
                row.clear();
                slot_index = 0;
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

void RIS(Ciphertext<DCRTPoly>& C, int p, int s, CryptoContext<DCRTPoly>& context) {
    Ciphertext<DCRTPoly> copyC;
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        copyC = context->EvalRotate(C, shift);
        C = context->EvalAdd(C, copyC);
    }
}

void RR(Ciphertext<DCRTPoly>& C, int p, int s, CryptoContext<DCRTPoly>& context) {
    Ciphertext<DCRTPoly> copyC;
    for (int i = 0; i < log2(s); i++) {
        int shift = - p * pow(2, i);
        copyC = context->EvalRotate(C, shift);
        C = context->EvalAdd(C, copyC);
    }
}

void PolyEval(Ciphertext<DCRTPoly>& R, Ciphertext<DCRTPoly>& I, double PA3, double PA2, double PA1, double PA0, CryptoContext<DCRTPoly>& context, double slot_count) {   
    Ciphertext<DCRTPoly> I2 = context->EvalMultAndRelinearize(I, I);
    Ciphertext<DCRTPoly> a3I = context->EvalMult(I, PA3);
    Ciphertext<DCRTPoly> I3 = context->EvalMultAndRelinearize(I2, a3I);

    I2 = context->EvalMult(I2, PA2);
    R = context->EvalAdd(I2, I3);
    I = context->EvalMult(I, PA1);

    size_t to_drop = R->GetLevel() - I->GetLevel();
    context->LevelReduceInPlace(I, nullptr, to_drop);
    
    R = context->EvalAdd(R, I);
    R = context->EvalAdd(R, PA0);
}

void PMultiply(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (uint i = 0; i < A.size(); i++) { C[i] = A[i] * B[i]; }
}

void PAddition(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (uint i = 0; i < A.size(); i++) { C[i] = A[i] + B[i]; }
}

void PSubtraction(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (uint i = 0; i < A.size(); i++) { C[i] = A[i] - B[i]; }
}

void PRIS(std::vector<double>& C, int p, int s) {
    for (uint i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        std::vector<double> copyC = C;
        rotate(C.begin(), C.begin() + shift, C.end());
        PAddition(C, C, copyC);
    }
}

void PRR(std::vector<double>& C, int p, int s) {
    for (uint i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        std::vector<double> copyC = C;
        rotate(C.rbegin(), C.rbegin() + shift, C.rend());
        PAddition(C, C, copyC);
    }
}

std::vector<double> RSigmoid(std::vector<double>& C) {
    std::vector<double> R(C.size());
    for (uint i = 0; i < C.size(); i++) { R[i] = 1.0 / (1.0 + std::exp(-C[i])); }
    return R;
}

std::vector<double> RSoftplus(std::vector<double>& C) {
    std::vector<double> R(C.size());
    for (uint i = 0; i < C.size(); i++) {
        if (C[i] > 20) R[i] = C[i];            
        if (C[i] < -20) R[i] = std::exp(C[i]);
        R[i] = std::log1p(std::exp(C[i]));  
    }
    return R;
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

    for (uint tt = 0; tt < packed_X_t.size(); tt++) {
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

int main(int argc, char* argv[]) {

    std::cout << WHITE << DIM << "PID: process " << ::getpid() << " (parent: " << ::getppid() << ")" << RESET << std::endl;

    std::cout << CYAN << BOLD << "\n===========================================================" << RESET << std::endl;
    std::cout << CYAN << BOLD << "       OpenFHE HOMOMORPHIC FINETUNING DEMO" << RESET << std::endl;
    std::cout << CYAN << BOLD << "===========================================================\n" << RESET << std::endl;

    // ---------------------------------------------------------
    // 1. Parameter Setup
    // ---------------------------------------------------------
    std::cout << MAGENTA << BOLD << "[1/6] INITIALIZING CONTEXT & KEYS..." << RESET << std::endl;
    auto start_init = std::chrono::steady_clock::now();

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetRingDim(1 << 16);
    parameters.SetSecurityLevel(HEStd_128_classic); 
    parameters.SetMultiplicativeDepth(24);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(57);

    SecretKeyDist secretKeyDist = SPARSE_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(6);

    CryptoContext<DCRTPoly> context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE); 

    std::cout << "  CKKS Ring Dimension: " << YELLOW << context->GetRingDimension() << RESET << std::endl;
    //std::cout << "  Multiplicative Depth: " << YELLOW << parameters.GetMultiplicativeDepth() << RESET << std::endl;

    auto keyPair = context->KeyGen();
    context->EvalMultKeyGen(keyPair.secretKey);

    std::vector<uint32_t> levelBudget = {3, 3};
    int slot_count = 32768; 

    context->EvalBootstrapSetup(levelBudget, {0, 0}, slot_count);
    context->EvalBootstrapKeyGen(keyPair.secretKey, slot_count);

    std::vector<int> key_index = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, -1, -2, -4, -8, -16, -32, -64, -128, -256};
    context->EvalRotateKeyGen(keyPair.secretKey, key_index);

    auto end_init = std::chrono::steady_clock::now();
    std::cout << GREEN << "  [+] Key Generation & Context Setup completed in " << std::chrono::duration<double>(end_init - start_init).count() << " seconds!" << RESET << "\n" << std::endl;

    // ---------------------------------------------------------
    // 2. Load Datasets & Model
    // ---------------------------------------------------------
    std::cout << MAGENTA << BOLD << "[2/6] LOADING DATASETS & MODEL..." << RESET << std::endl;
    auto start_io = std::chrono::steady_clock::now();

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
    bool paths_envvar = (args.find("paths_envvar") != args.end());

    std::cout << "  [*] Training size: " << len_X_train << " | Epochs: " << round << " | LR: " << LR << "\n";

    double number_of_items_in_ctext = ceil(slot_count / (h[1] * h[2]));
    int model_repeat = slot_count / (h[1] * h[2]); 
    double b = number_of_items_in_ctext / model_repeat; 
    double m = ceil(len_X_train / number_of_items_in_ctext); 
    // double m_test = ceil(len_X_test / number_of_items_in_ctext); 

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

    std::vector<double> m1(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i % h[2] == 0) m1[i] = 1; }
    std::vector<double> m2(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i%(h[1] * h[2]) < h[2]) m2[i] = 1; }
    std::vector<double> m3(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i%(h[2]) == 0) m3[i] = 1; }
    std::vector<double> m4(slot_count, 0); for (int i = 0; i < slot_count; i++) { if (i%(h[2] * h[3]) < h[4] && i%(h[1] * h[2]) < h[2] * h[3]) m4[i] = 1; }

    std::vector<Plaintext> P_W;
    for (int i = 0; i < ell; i++) {
        Plaintext P_W_i = context->MakeCKKSPackedPlaintext(packed_W[i]);
        P_W.push_back(P_W_i);
    }
    Ciphertext<DCRTPoly> C_W_3 = context->Encrypt(keyPair.publicKey, P_W[3]);

    auto end_io = std::chrono::steady_clock::now();
    std::cout << GREEN << "  [+] Data structures loaded & prep in " << std::chrono::duration<double>(end_io - start_io).count() << " seconds!" << RESET << "\n" << std::endl;

    // ---------------------------------------------------------
    // 3. Encrypting Client Data
    // ---------------------------------------------------------
    std::cout << MAGENTA << BOLD << "[3/6] ENCRYPTING DATA (CLIENT-SIDE)..." << RESET << std::endl;
    
    std::cout << "  [*] Encrypting Finetuning Samples (" << m << " batches)...\n";
    std::vector<Ciphertext<DCRTPoly>> C_X, C_y;
    auto start_encryption = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < m; i++) {
        printProgress((double)(i + 1) / m);
        Plaintext P_X_i = context->MakeCKKSPackedPlaintext(packed_X[i]);
        Plaintext P_y_i = context->MakeCKKSPackedPlaintext(packed_y[i]);
        C_X.push_back(context->Encrypt(keyPair.publicKey, P_X_i));
        C_y.push_back(context->Encrypt(keyPair.publicKey, P_y_i));
    }
    std::cout << std::endl;

    // std::cout << "  [*] Encrypting Test Samples (" << m_test << " batches)...\n";
    // std::vector<Ciphertext<DCRTPoly>> C_X_t;
    // for(int i = 0; i < m_test; i++) {
    //     printProgress((double)(i + 1) / m_test);
    //     Plaintext P_X_t_i = context->MakeCKKSPackedPlaintext(packed_X_t[i]);
    //     C_X_t.push_back(context->Encrypt(keyPair.publicKey, P_X_t_i));
    // }
    // std::cout << std::endl;

    auto end_encryption = std::chrono::high_resolution_clock::now();
    std::cout << GREEN << "  [+] Datasets encrypted in " << std::chrono::duration<double>(end_encryption - start_encryption).count() << " seconds!" << RESET << "\n" << std::endl;

    // ---------------------------------------------------------
    // 4. Pre-Evaluation
    // ---------------------------------------------------------
    std::cout << MAGENTA << BOLD << "[4/6] PRE-EVALUATION (BEFORE FINETUNING)..." << RESET << std::endl;
    EvalResults before = evaluate_model_ptext(packed_X_t, y_t, packed_W, chosen_indices, slot_count, h, m1, m2, m3, m4, len_X_test, number_of_items_in_ctext, "Before Fine-tuning");

    // ---------------------------------------------------------
    // 5. Homomorphic Finetuning
    // ---------------------------------------------------------
    std::cout << MAGENTA << BOLD << "\n[5/6] HOMOMORPHIC FINETUNING (SERVER-SIDE)..." << RESET << std::endl;
    double total_training_time = 0.0;

    for (int rn = 0; rn < round; rn++) {
        std::cout << BLUE << BOLD << "  [=== Epoch " << rn + 1 << " / " << int(round) << " ===]" << RESET << std::endl;

        for (int t = 0; t < m; t++) {
            const auto start_batch = std::chrono::steady_clock::now();
        
            std::vector<double> all_zeros(slot_count, 0);
            Plaintext all_zeros_plain = context->MakeCKKSPackedPlaintext(all_zeros);
            Ciphertext<DCRTPoly> d_W_4 = context->Encrypt(keyPair.publicKey, all_zeros_plain);
            
            std::cout << "\r    " << YELLOW << "Working ->" << RESET << " Batch " << t + 1 << " / " << m << std::flush;
            
            for (int item = 0; item < b; item++) {
                if (t*b+item < len_X_train) {
                    Ciphertext<DCRTPoly> L_0 = C_X[t*b+item];
                    
                    // ---- Layer 1 ----
                    Ciphertext<DCRTPoly> U_1 = context->EvalMult(L_0, P_W[0]);
                    RIS(U_1, 1, h[0], context);
                    Plaintext P_m1 = context->MakeCKKSPackedPlaintext(m1);
                    U_1 = context->EvalMult(U_1, P_m1);
                    RR(U_1, 1, h[2], context);

                    Ciphertext<DCRTPoly> L_1;
                    PolyEval(L_1, U_1, -2.38253701e-05,  3.51409155e-03,  8.49230659e-01,  1.81535534e+00, context, slot_count);

                    // ---- Layer 2 ----
                    Ciphertext<DCRTPoly> U_2 = context->EvalMult(L_1, P_W[1]);
                    RIS(U_2, h[2], h[1], context);
                    Plaintext P_m2 = context->MakeCKKSPackedPlaintext(m2);
                    U_2 = context->EvalMult(U_2, P_m2);
                    RR(U_2, h[2], h[3], context);

                    Ciphertext<DCRTPoly> L_2;
                    PolyEval(L_2, U_2, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, context, slot_count);

                    // ---- Layer 3 ----
                    Ciphertext<DCRTPoly> U_3 = context->EvalMult(L_2, P_W[2]);
                    RIS(U_3, 1, h[2], context);
                    Plaintext P_m3 = context->MakeCKKSPackedPlaintext(m3);
                    U_3 = context->EvalMult(U_3, P_m3);
                    RR(U_3, 1, h[4], context);

                    Ciphertext<DCRTPoly> L_3;
                    PolyEval(L_3, U_3, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, context, slot_count);

                    // ---- Layer 4 ----
                    Ciphertext<DCRTPoly> U_4 = context->EvalMultAndRelinearize(L_3, C_W_3);
                    RIS(U_4, h[2], h[3], context);
                    Plaintext P_m4 = context->MakeCKKSPackedPlaintext(m4);
                    U_4 = context->EvalMult(U_4, P_m4);
                    
                    Ciphertext<DCRTPoly> L_4;
                    PolyEval(L_4, U_4, 5.60468547e-06, -8.37716501e-04, 3.67742962e-02, 5.37938314e-01, context, slot_count);
                  
                    // ****************** Backpropagation ****************** //
                    Ciphertext<DCRTPoly> E_4 = context->EvalSub(C_y[t*b+item], L_4);
                    P_m4 = context->MakeCKKSPackedPlaintext(m4);
                    E_4 = context->EvalMult(E_4, P_m4);
                    RR(E_4, h[2], h[3], context);

                    Ciphertext<DCRTPoly> interm = context->EvalMultAndRelinearize(E_4, L_3);
                    d_W_4 = context->EvalAdd(d_W_4, interm);
                }
            }

            std::vector<double> eta(slot_count, LR/(b*number_of_items_in_ctext));
            Plaintext eta_P = context->MakeCKKSPackedPlaintext(eta); 

            Ciphertext<DCRTPoly> d_W_4_scaled = context->Encrypt(keyPair.publicKey, all_zeros_plain);       
            d_W_4 = context->EvalMult(d_W_4, eta_P);
            RIS(d_W_4, h[1]*h[2], model_repeat, context);
            d_W_4_scaled = context->EvalAdd(d_W_4, d_W_4_scaled);
    
            C_W_3 = context->EvalAdd(C_W_3, d_W_4_scaled);
            
            std::cout << "\r    " << CYAN << "Bootstrapping..." << RESET << std::string(10, ' ') << std::flush;
            Ciphertext<DCRTPoly> C_W_b3 = context->EvalBootstrap(C_W_3);
            C_W_3 = C_W_b3;

            const auto finish_batch = std::chrono::steady_clock::now();
            double batch_time = std::chrono::duration<double>(finish_batch - start_batch).count();
            total_training_time += batch_time;
            
            std::cout << "\r    " << GREEN << "-> Batch " << t + 1 << " completed in " << std::fixed << std::setprecision(2) << batch_time << " seconds." << RESET << std::string(15, ' ') << "\n";
        }
    }

    std::cout << GREEN << BOLD << "\n  [+] Finetuning finished in " << total_training_time << " seconds!" << RESET << std::endl;

    // ---------------------------------------------------------
    // 6. Post-Evaluation
    // ---------------------------------------------------------
    std::cout << MAGENTA << BOLD << "\n[6/6] POST-EVALUATION (AFTER FINETUNING)..." << RESET << std::endl;
    
    // Decrypt C_W[3] and update the weights locally for inference
    Plaintext P_W_3;
    context->Decrypt(keyPair.secretKey, C_W_3, &P_W_3);
    P_W_3->SetLength(slot_count);
    std::vector<double> W_3 = P_W_3->GetRealPackedValue();
    packed_W[3] = W_3;

    EvalResults after = evaluate_model_ptext(packed_X_t, y_t, packed_W, chosen_indices, slot_count, h, m1, m2, m3, m4, len_X_test, number_of_items_in_ctext, "After Fine-tuning");
   
    double improvement = after.accuracy_subset - before.accuracy_subset;
    std::cout << BLUE << BOLD << "\n===========================================================" << RESET << std::endl;
    std::cout << BLUE << BOLD << "  [!] SUBSET ACCURACY IMPROVEMENT: " << (improvement > 0 ? GREEN : RED) << BOLD << "+" << std::fixed << std::setprecision(2) << improvement << "%" << RESET << std::endl;
    std::cout << BLUE << BOLD << "===========================================================\n" << RESET << std::endl;

    return 0;
}
