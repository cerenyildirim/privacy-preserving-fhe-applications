#include "heongpu.cuh"
#include "../example_util.h"
#include <chrono>
#include <stdlib.h>
#include <signal.h>
// --------------------------------------------------------------
// File:    credit.cu
// Author:  Ceren Yıldırım
// Created: 2025-08-25
// --------------------------------------------------------------

void sig_handler(int sig) { // signal handler for shell signals
    
    std::cerr << "Signal ("<< sig <<") interrupted normal execution" << std::endl;
    abort();
    exit(EXIT_FAILURE);
}

// Set to 1 to enable debug messages (outputs the multiplicative depth of the ciphertexts)
#define DEBUG_DEPTH      1
#define DEBUG_LOG(domain, msg) \
    do { if (domain) std::cout << msg << std::endl; } while(0)

struct EvalResults {
    double accuracy_full;     // accuracy over full test set
    double accuracy_subset;   // accuracy over chosen subset
    int correct_full;         // number of correct predictions (full set)
    int correct_subset;       // number of correct predictions (subset)
    int total_full;           // total samples in full set
    int total_subset;         // total samples in subset
};

// Decyrpts and prints a ciphertext
void printCiphertext(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEDecryptor<heongpu::Scheme::CKKS>& decryptor, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder) {
    heongpu::Plaintext<heongpu::Scheme::CKKS> X_b(context);
    std::vector<double> X_b_vec;
    decryptor.decrypt(X_b, C);
    encoder.decode(X_b_vec, X_b);
    display_vector(X_b_vec, 4UL, 5);
}

// Function to read the training samples and labels (in batched form for homomorphic processing)
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

// Function to read the test samples and labels (in regular plaintext form)
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

// Function to read the model weights (in batched form for homomorphic processing)
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

// Function to read the indices of the test samples used to measure the accuracy of the finetuning
std::vector<int> readChosenIndices(const std::string& filename) {
    std::ifstream file(filename);
    int idx;
    std::vector<int> indices;
    while (file >> idx) {
        indices.push_back(idx);
    }
    return indices;
}

// Rotate For Inner Sum (RIS(c,p,s)) is used to compute the inner-sum of a packed vector c by homomorphically rotating it to the left with RotL(c,p) and by adding it to itself iteratively log2(s) times
void RIS(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, int p, int s, heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> copyC(context);
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        operators.rotate_rows(C, copyC, galois_key, shift); 
        operators.add_inplace(C, copyC);
       
    }
}

// Rotate For Replication (RR(c,p,s)) replicates the values in the slots of a ciphertext by rotating the ciphertext to the right with RotR(c,p) and by adding to itself, iteratively log2(s) times
void RR(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, int p, int s, heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> copyC(context);
    for (int i = 0; i < log2(s); i++) {
        int shift = - p * pow(2, i);
        operators.rotate_rows(C, copyC, galois_key, shift); 
        operators.add_inplace(C, copyC);
    }
}

// Evaluates the polynomial approximation of the softplus function homomorphically
void Softplus(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS>& I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS> &relin_key, double scale, double slot_count, heongpu::HEDecryptor<heongpu::Scheme::CKKS>& decryptor) {   

    // printCiphertext(I, context, decryptor, encoder);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I2(context);
    operators.multiply(I, I, I2); // I2 Depth 3
    operators.relinearize_inplace(I2, relin_key);
    operators.rescale_inplace(I2);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> a3I(context);

    operators.multiply_plain(I, PA3, a3I, scale); // a3I Depth 3
    operators.rescale_inplace(a3I);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I3(context);
    operators.multiply(I2, a3I, I3); // I3 Depth 4
    operators.relinearize_inplace(I3, relin_key);
    operators.rescale_inplace(I3);

    operators.multiply_plain_inplace(I2, PA2, scale); // I2 Depth 4
    operators.rescale_inplace(I2);

    while (I2.depth() < I3.depth()) { // unnecessary
        operators.mod_drop_inplace(I2);
    }

    operators.add(I3, I2, R); // R Depth 4

    operators.multiply_plain_inplace(I, PA1, scale); // I Depth 3
    operators.rescale_inplace(I);

    while (I.depth() < R.depth()) {
        operators.mod_drop_inplace(I);
    }
    operators.add_inplace(R, I);

    operators.add_plain_inplace(R, PA0);
}

// Evaluates the polynomial approximation of the sigmoid function homomorphically
//void Sigmoid(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS> I, heongpu::Plaintext<heongpu::Scheme::CKKS> PA3, heongpu::Plaintext<heongpu::Scheme::CKKS> PA2, heongpu::Plaintext<heongpu::Scheme::CKKS> PA1, heongpu::Plaintext<heongpu::Scheme::CKKS> PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key, double scale, double slot_count) {
void Sigmoid(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS>& I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key, double scale, double slot_count) {

    heongpu::Ciphertext<heongpu::Scheme::CKKS> I2(context);
    operators.multiply(I, I, I2);
    operators.relinearize_inplace(I2, relin_key);
    operators.rescale_inplace(I2);

    // std::cout << "I2 depth: " << I2.depth() << std::endl;

    heongpu::Ciphertext<heongpu::Scheme::CKKS> a3I(context);

    operators.multiply_plain(I, PA3, a3I, scale); // a3I Depth 3
    operators.rescale_inplace(a3I);
    heongpu::Ciphertext<heongpu::Scheme::CKKS> I3(context);
    operators.multiply(I2, a3I, I3); // I3 Depth 4
    operators.relinearize_inplace(I3, relin_key);
    operators.rescale_inplace(I3);

    // std::cout << "I3 depth: " << I3.depth() << std::endl;

    operators.multiply_plain_inplace(I2, PA2, scale); // I2 Depth 9
    operators.rescale_inplace(I2);

    while (I2.depth() < I3.depth()) {
        operators.mod_drop_inplace(I2); // Ensure I2 Depth matches I3 Depth
    }
    operators.add(I3, I2, R); // R Depth 10

    operators.multiply_plain_inplace(I, PA1, scale); // I Depth 9
    operators.rescale_inplace(I);

    while (I.depth() < R.depth()) {
        operators.mod_drop_inplace(I); // Ensure I Depth matches R Depth
    }
    operators.add_inplace(R, I);

    operators.add_plain_inplace(R, PA0); // R Depth 10
}

void PMultiply(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] * B[i];
    }
}

// Plaintext addition operation for inferring test samples
void PAddition(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] + B[i];
    }
}

// Plaintext subtraction operation for inferring test samples
void PSubtraction(std::vector <double>& C, std::vector<double>& A, std::vector<double>& B) {
    for (int i = 0; i < A.size(); i++) {
        C[i] = A[i] - B[i];
    }
}

// Plaintext Rotate For Inner SUm operation for inferring test samples
void PRIS(std::vector<double>& C, int p, int s) {
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        std::vector<double> copyC = C;
        rotate(C.begin(), C.begin() + shift, C.end());
        PAddition(C, C, copyC);
    }
}

// Plaintext Rotate for Replication operation for inferring test samples
void PRR(std::vector<double>& C, int p, int s) {
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        std::vector<double> copyC = C;
        rotate(C.rbegin(), C.rbegin() + shift, C.rend());
        PAddition(C, C, copyC);
    }
}

// Sigmoid function
std::vector<double> RSigmoid(std::vector<double>& C) {
    std::vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        R[i] = 1.0 / (1.0 + std::exp(-C[i]));
    }
    return R;
}

// Softplus function
std::vector<double> RSoftplus(std::vector<double>& C) {
    std::vector<double> R(C.size());
    for (int i = 0; i < C.size(); i++) {
        if (C[i] > 20) 
            R[i] = C[i];            
        if (C[i] < -20) 
            R[i] = std::exp(C[i]);
        R[i] = std::log1p(std::exp(C[i]));  
    }
    return R;
}

EvalResults evaluate_model(
    std::vector<std::vector<double>>& packed_X_t,
    std::vector<std::vector<double>>& y_t,
    std::vector<std::vector<double>>& packed_W,
    std::vector<int>& chosen_indices,
    int slot_count,
    std::vector<int>& h,
    std::vector<double>& m1,
    std::vector<double>& m2,
    std::vector<double>& m3,
    std::vector<double>& m4,
    const std::string& title = "Evaluation Results"
) {
    int corr = 0;
    int corr_selected = 0;

    std::cout << "\n=== " << title << " ===\n";
    std::cout << "Testing starts (" << packed_X_t.size() << " samples)...\n";

    for (int tt = 0; tt < packed_X_t.size(); tt++) {
        if (tt % 100 == 0) 
            std::cout << "  Progress: " << tt << "/" << packed_X_t.size() << "\n";

        std::vector<double> L_0 = packed_X_t[tt];

        // ---- Layer 1 ----
        std::vector<double> U_1(slot_count);
        PMultiply(U_1, L_0, packed_W[0]);
        PRIS(U_1, 1, h[0]);
        PMultiply(U_1, U_1, m1);
        PRR(U_1, 1, h[2]);
        std::vector<double> L_1 = RSoftplus(U_1);

        // ---- Layer 2 ----
        std::vector<double> U_2(slot_count);
        PMultiply(U_2, L_1, packed_W[1]);
        PRIS(U_2, h[2], h[1]);
        PMultiply(U_2, U_2, m2);
        PRR(U_2, h[2], h[3]);
        std::vector<double> L_2 = RSoftplus(U_2);

        // ---- Layer 3 ----
        std::vector<double> U_3(slot_count);
        PMultiply(U_3, L_2, packed_W[2]);
        PRIS(U_3, 1, h[2]);
        PMultiply(U_3, U_3, m3);
        PRR(U_3, 1, h[4]);
        std::vector<double> L_3 = RSoftplus(U_3);

        // ---- Layer 4 ----
        std::vector<double> U_4(slot_count);
        PMultiply(U_4, L_3, packed_W[3]);
        PRIS(U_4, h[2], h[3]);
        PMultiply(U_4, U_4, m4);
        std::vector<double> L_4 = RSigmoid(U_4);

        int y_pred = (L_4[0] > 0.5) ? 1 : 0;
        int y_true = int(y_t[tt][0]);

        if (y_pred == y_true) {
            corr++;
            if (std::find(chosen_indices.begin(), chosen_indices.end(), tt) != chosen_indices.end())
                corr_selected++;
        }
    }

    int total_full = packed_X_t.size();
    int total_subset = chosen_indices.size();

    double acc_full = 100.0 * corr / total_full;
    double acc_subset = 100.0 * corr_selected / total_subset;

    // --- Print results ---
    std::cout << "\n=== Test Results ===\n";
    std::cout << "Full Test Set:\n";
    std::cout << "  Correct predictions: " << corr
              << " / " << total_full
              << " (" << std::fixed << std::setprecision(2)
              << acc_full << "% accuracy)\n\n";

    std::cout << "Subset (Finetuning-like Samples):\n";
    std::cout << "  Correct predictions: " << corr_selected
              << " / " << total_subset
              << " (" << std::fixed << std::setprecision(2)
              << acc_subset << "% accuracy)\n\n";

    return {acc_full, acc_subset, corr, corr_selected, total_full, total_subset};
}

// https://stackoverflow.com/a/63360252/8014672
std::string get_env(const char* key) {
    if (key == nullptr) {
        throw std::invalid_argument("Null pointer passed as environment variable name");
    }
    if (*key == '\0') {
        throw std::invalid_argument("Value requested for the empty-name environment variable");
    }
    const char* ev_val = getenv(key);
    if (ev_val == nullptr) {
        throw std::runtime_error("Environment variable not defined");
    }
    return std::string { ev_val };
}


int main(int argc, char* argv[])
{

 /*
  signal (SIGKILL, sig_handler);
  signal (SIGTERM, sig_handler);
  signal (SIGINT, sig_handler);
  */
  std::cout << "PID: process " << ::getpid() << " (parent: " << ::getppid() << ")" << std::endl;
  try {
    int defaultcudaDevice = 0;

    // ********************************************************************** //
    // *************************** Start FINETUNING ************************* //
    // ********************************************************************** //

    double len_X_train = 2000; // The size of the training (finetuning) set
    double len_X_test = 4500; // The size of teh test set
    double ell = 4; // The number of layers in the neural network
    double LR = 0.05; // The learning rate of SGD
    double round = 1; // The number of epochs 
    std::vector<int> h = {32, 64, 32, 16, 1}; // The number of nodes in each neural network layer (i.e., d, h_1, h_2, h_3, h_4)
    std::string data_dir = "./txts";   // default folder
        
    // Parse arguments of the form --train_size=... --epochs=... --learning_rate=... --data_dir=...
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

    // Update values from args if provided
    if (args.find("train_size") != args.end()) len_X_train = std::stoi(args["train_size"]);
    if (args.find("epochs") != args.end()) round = std::stoi(args["epochs"]);
    if (args.find("learning_rate") != args.end()) LR = std::stod(args["learning_rate"]);
    if (args.find("data_dir") != args.end()) data_dir = args["data_dir"];
    if (args.find("gpu_device") != args.end()) defaultcudaDevice = std::stoi(args["gpu_device"]);
    
    bool paths_envvar = (args.find("paths_envvar") != args.end());

    cudaSetDevice(defaultcudaDevice); // Use it for memory pool
    std::cout << "Utilized Cuda Device : " << defaultcudaDevice <<std::endl;

    // Initialize encryption parameters for the CKKS scheme
    heongpu::HEContext<heongpu::Scheme::CKKS> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::sec128); 

    size_t poly_modulus_degree = 65536;
    const int slot_count = poly_modulus_degree / 2;
    context.set_poly_modulus_degree(poly_modulus_degree);

    // A maximum of 1761 bits coefficient modulus can be used for the polynomial modulus degree of 65536
    int number_of_moduli = 30;
    int scale_bits = 50;
    std::vector<int> moduli(number_of_moduli, scale_bits);
    moduli.insert(moduli.begin(), 60);
    context.set_coeff_modulus_bit_sizes(moduli,
       {60, 60, 60}); 
    context.generate();
    context.print_parameters();

    std::cout << "moduli" << ": ";
    for (int v : moduli) std::cout << v << " ";
        std::cout << "\n";

    // The scale is set to 2^50, resulting in 50 bits of precision before the decimal point
    double scale = pow(2.0, scale_bits);

    // Generate keys: the public key for encryption, the secret key for decryption and evaluation key (relinkey) for relinearization
    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(
        context,
        16); // Hamming weight is 16 in this example
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
    // Generates all bootstrapping parameters before bootstrapping
    operators.generate_bootstrapping_params(scale, boot_config, heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);

    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key_rot(context, 14);
    keygen.generate_galois_key(galois_key_rot, secret_key,
        heongpu::ExecutionOptions().set_storage_type(heongpu::storage_type::DEVICE));
    std::cout << "Galois key generation finished!" << std::endl;

    // This portion about arg parse has been moved at beginning of main // arsalan

    // Print model and training set inormation
    std::cout << "Training set size: " << len_X_train << "\n";
    std::cout << "Number of epochs: " << round << "\n";
    std::cout << "Learning rate: " << LR << "\n";

    std::cout << "Model summary: " << "\n";
    std::cout << "\tLayer 1: " << h[1] << " nodes\n";
    std::cout << "\tLayer 2: " << h[2] << " nodes\n";
    std::cout << "\tLayer 3: " << h[3] << " nodes\n";
    std::cout << "\tLayer 4: " << h[4] << " nodes\n";

    double number_of_items_in_ctext = ceil(slot_count / (h[1] * h[2]));

    std::cout << "The # of items in a ciphertext: " << number_of_items_in_ctext << std::endl;

    int model_repeat = slot_count / (h[1] * h[2]); // the number of times the model is repeated in a ciphertext, equal to number_of_items_in_ctext
    double b = number_of_items_in_ctext / model_repeat; // number of batches stored in a ciphertext
    double m = ceil(len_X_train / number_of_items_in_ctext); // the number of ciphertexts (generally referring to a single batch, when b=1)
    std::cout << "The # of batches (m): " << m << std::endl;
    std::cout << "The # of batches stored in a ciphertext (b): " << b << std::endl;

    // Read the training set, model, and test set values
    std::vector<std::vector<double>> packed_X(m, std::vector<double>(slot_count));
    std::vector<std::vector<double>> packed_X_t(len_X_test, std::vector<double>(h[1]*h[2]));
    std::vector<std::vector<double>> packed_y(m, std::vector<double>(slot_count));
    std::vector<std::vector<double>> y_t(len_X_test, std::vector<double>(1)); 
    std::vector<std::vector<double>> packed_W(ell, std::vector<double>(slot_count));
    std::vector<int> chosen_indices;

    if (paths_envvar) std::cout << "++++++paths_ennvar provided" << std::endl; 

    std::cout << "--> " <<  data_dir + "/packed_X.txt" << std::endl; 
    
    packed_X = read2DArrayTrain(  paths_envvar ? get_env("PACKED_X") : data_dir + "/packed_X.txt", slot_count);
    packed_X_t = read2DArrayTest( paths_envvar ? get_env("PACKED_X_T") :  data_dir + "/packed_X_t.txt");
    packed_y = read2DArrayTrain(  paths_envvar ? get_env("PACKED_Y") :   data_dir + "/packed_y.txt", slot_count);
    y_t = read2DArrayTest(  paths_envvar ? get_env("Y_TEST") :  data_dir + "/y_test.txt");
    packed_W = read2DArrayModel( paths_envvar ? get_env("PACKED_W") :   data_dir + "/packed_W.txt", slot_count, h);
    chosen_indices = readChosenIndices(  paths_envvar ? get_env("CHOSEN_INDICES") :  data_dir + "/chosen_indices.txt");

    // Generate masks for each layer to be used during homomorphic finetuning
    std::vector<double> m1(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i % h[2] == 0)
            m1[i] = 1;
    }

    std::vector<double> m2(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[1] * h[2]) < h[2])
            m2[i] = 1;
    }

    std::vector<double> m3(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[2]) == 0)
            m3[i] = 1;
    }

    std::vector<double> m4(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[2] * h[3]) < h[4]) {
            if (i%(h[1] * h[2]) < h[2] * h[3])
                m4[i] = 1;
        }
    }
 
    // Masks for the incomplete batch ciphertexts
    int full_slots = (int(len_X_train) % (int(slot_count) / int(h[1] * h[2])));
    std::vector<double> m1_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i % h[2] == 0)
        m1_i[i] = 1;
    }

    std::vector<double> m2_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i%(h[1] * h[2]) < h[2])
            m2_i[i] = 1;
    }

    std::vector<double> m3_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i%(h[2]) == 0)
            m3_i[i] = 1;
    }

    std::vector<double> m4_i(slot_count, 0);
    for (int i = 0; i < full_slots*(h[1]*h[2]); i++) {
        if (i%(h[1] * h[2]) < h[4])
            m4_i[i] = 1;
    }

    // Encode P_W (model weights)
    std::vector<heongpu::Plaintext<heongpu::Scheme::CKKS>> P_W;
    for (int i = 0; i < ell; i++) {
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_W_i(context);
            encoder.encode(P_W_i, packed_W[i], scale);
            P_W.emplace_back(std::move(P_W_i));
    }
    
    // Encode softplus coefficients
    //SP_C3 -2.38253701e-05
    //SP_C2 3.51409155e-03
    //SP_C1 8.49230659e-01
    //SP_C0 1.81535534e+00

    // Encode sigmoid coefficients
    // SG_C3  5.60468547e-06
    // SG_C2  -8.37716501e-04
    // SG_C1  3.67742962e-02
    // SG_C0  5.37938314e-01

    std::cout << "Encoding finished!" << std::endl;

    // **** Encrypt the model **** //
    // Encrypting the model
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> C_W;

    for (int i = 0; i < ell; i++) {
        heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_i(context);
        encryptor.encrypt(C_W_i, P_W[i]);
        C_W.emplace_back(std::move(C_W_i));
    }

    std::cout << "Encryption finished! " << ell << std::endl;

    // After training but before fine-tuning
    EvalResults before = evaluate_model(packed_X_t, y_t, packed_W, chosen_indices,
                                    slot_count, h, m1, m2, m3, m4,
                                    "Before Fine-tuning");

    // **********   TRAINING (FINETUNING) STARTS HERE   ********** //

    std::cout << "=== Training (finetuning) starts! ===" << std::endl;

    double total_training_time = 0.0;

    for (int rn = 0; rn < round; rn++) {
        for (int t = 0; t < m; t++) {
            std::cout << " >>> Round " << rn << " - Batch " << t << std::endl;

            const auto start_prep{std::chrono::steady_clock::now()};
        
            // Initialize the d_W_4 for the new batch
            std::vector<double> all_zeros(slot_count, 0);
            heongpu::Plaintext<heongpu::Scheme::CKKS> all_zeros_plain(context);
            encoder.encode(all_zeros_plain, all_zeros, scale);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_4(context);
            encryptor.encrypt(d_W_4, all_zeros_plain);
            
            const auto start_clients{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_for_prep{(start_clients - start_prep)};
            
            for (int item = 0; item < b; item++) {
                if (t*b+item < len_X_train) {
                    heongpu::Plaintext<heongpu::Scheme::CKKS> L_0(context);
                    encoder.encode(L_0, packed_X[t*b+item], scale);
                    
                    // ---- Layer 1 ----

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);
                    
                    operators.multiply_plain(C_W[0], L_0, U_1); 
                    operators.rescale_inplace(U_1);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_1 after multiply: " << U_1.depth());

                    RIS(U_1, 1, h[0], galois_key_rot, operators, context);

                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1;
                    encoder.encode(P_m1, m1, scale);
                    
                    while (P_m1.depth() < U_1.depth()) {
                        operators.mod_drop_inplace(P_m1); 
                    }
                    
                    operators.multiply_plain_inplace(U_1, P_m1);
                    operators.rescale_inplace(U_1);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth of U_1 after RIS: " << U_1.depth());

                    RR(U_1, 1, h[2], galois_key_rot, operators, context);

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);

                    Softplus(L_1, U_1, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth L_1 after softplus: " << L_1.depth());

                    // ---- Layer 2 ----

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);
    
                    while (L_1.depth() > C_W[1].depth()) {
                        operators.mod_drop_inplace(C_W[1]); 
                    }
        
                    operators.multiply(L_1, C_W[1], U_2);
                    operators.relinearize_inplace(U_2, relin_key);
                    operators.rescale_inplace(U_2);
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after multiply: " << U_2.depth());

                    RIS(U_2, h[2], h[1], galois_key_rot, operators, context);
                
                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
                    encoder.encode(P_m2, m2, scale);
                    while (U_2.depth() > P_m2.depth()) {
                        operators.mod_drop_inplace(P_m2); 
                    }
            
                    operators.multiply_plain_inplace(U_2, P_m2); 
                    operators.rescale_inplace(U_2);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after RIS: " << U_2.depth());     
                    
                    RR(U_2, h[2], h[3], galois_key_rot, operators, context);

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
                    Softplus(L_2, U_2, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth L_2 after softplus: " << L_2.depth());

                    // ---- Layer 3 ----

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> U_3(context);

                    while (L_2.depth() > C_W[2].depth()) {
                        operators.mod_drop_inplace(C_W[2]); 
                    }
                       
                    operators.multiply(L_2, C_W[2], U_3);
                    operators.relinearize_inplace(U_3, relin_key);
                    operators.rescale_inplace(U_3);
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_3 after multiply: " << U_3.depth());

                    RIS(U_3, 1, h[2], galois_key_rot, operators, context);
                    
                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_m3;
                    encoder.encode(P_m3, m3, scale);
                    while (U_3.depth() > P_m3.depth()) {
                        operators.mod_drop_inplace(P_m3); 
                    }
                    
                    operators.multiply_plain_inplace(U_3, P_m3); 
                    // encoder.encode(P_m3, m3, scale); 
                    operators.rescale_inplace(U_3);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_3 after RIS: " << U_3.depth());     
                    
                    RR(U_3, 1, h[4], galois_key_rot, operators, context);

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> L_3(context);
                    Softplus(L_3, U_3, -2.38253701e-05, 3.51409155e-03, 8.49230659e-01, 1.81535534e+00, operators, context, encoder, relin_key, scale, slot_count, decryptor);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth L_3 after softplus: " << L_3.depth());

                    // ---- Layer 4 ----

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> U_4(context);

                    while (L_3.depth() > C_W[3].depth()) {
                        operators.mod_drop_inplace(C_W[3]); 
                    }

                    while (C_W[3].depth() > L_3.depth()) {
                        operators.mod_drop_inplace(L_3); 
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W[3]: " << C_W[3].depth());
                    
                    operators.multiply(L_3, C_W[3], U_4);
                    operators.relinearize_inplace(U_4, relin_key);
                    operators.rescale_inplace(U_4);
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_4 after multiply: " << U_4.depth());

                    RIS(U_4, h[2], h[3], galois_key_rot, operators, context);
                 
                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_m4;
                    encoder.encode(P_m4, m4, scale);

                    while (U_4.depth() > P_m4.depth()) {
                        operators.mod_drop_inplace(P_m4); 
                    }
                
                    while (P_m4.depth() > U_4.depth()) {
                        operators.mod_drop_inplace(U_4); 
                    }
                    
                    operators.multiply_plain_inplace(U_4, P_m4); 
                    operators.rescale_inplace(U_4);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_4 after RIS: " << U_4.depth());    
                    
                    heongpu::Ciphertext<heongpu::Scheme::CKKS> L_4(context);
                    Sigmoid(L_4, U_4, 5.60468547e-06, -8.37716501e-04, 3.67742962e-02, 5.37938314e-01, operators, context, encoder, relin_key, scale, slot_count);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth L_4 after sigmoid: " << L_4.depth());
                    
                    // while (L_4.depth() < (number_of_moduli - StoC_piece)) {
                    //     operators.mod_drop_inplace(L_4);
                    // }

                    // DEBUG_LOG(DEBUG_DEPTH, " > Depth L_4 before bootstrapping: " << L_4.depth());
            
                    // L_4 = operators.slim_bootstrapping(L_4, galois_key_rot, relin_key);

                    // DEBUG_LOG(DEBUG_DEPTH, " > Depth L_4 after bootstrapping: " << L_4.depth());

                    // ****************** Backpropagation ****************** //
                    DEBUG_LOG(DEBUG_DEPTH, " >> Backpropagation starts!");

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> E_4(context);
                    operators.negate(L_4, E_4); 
                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_y_curr(context);
                    encoder.encode(P_y_curr, packed_y[t*b+item], scale);

                    while (P_y_curr.depth() < E_4.depth()) {
                        operators.mod_drop_inplace(P_y_curr);
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth P_y_curr after mod drop: " << P_y_curr.depth());
                    
                    operators.add_plain_inplace(E_4, P_y_curr);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_4 after add: " << E_4.depth());
     
                    
                    while (E_4.depth() < P_m4.depth()) {
                        operators.mod_drop_inplace(E_4);
                    }

                    while (E_4.depth() > P_m4.depth()) {
                        operators.mod_drop_inplace(P_m4);
                    }

                   
                    operators.multiply_plain_inplace(E_4, P_m4); 
                    operators.rescale_inplace(E_4);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_4 after multiply: " << E_4.depth());


                    RR(E_4, h[2], h[3], galois_key_rot, operators, context);

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> interm(context);

                    while (L_3.depth() < E_4.depth()) {
                        operators.mod_drop_inplace(L_3); 
                    }

                    while (L_3.depth() > E_4.depth()) {
                        operators.mod_drop_inplace(E_4); 
                    }

                    operators.multiply(E_4, L_3, interm); 
                    operators.relinearize_inplace(interm, relin_key);
                    operators.rescale_inplace(interm);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth of interm after multiply: " << interm.depth());

                    
                    while (d_W_4.depth() < interm.depth()) {
                        operators.mod_drop_inplace(d_W_4);
                    }
                
                    while (d_W_4.depth() > interm.depth()) {
                        operators.mod_drop_inplace(interm); 
                    }
                
                    operators.add_inplace(d_W_4, interm);     

                }
            }

            const auto finish_clients{std::chrono::steady_clock::now()};

            const std::chrono::duration<double> elapsed_seconds_for_training{(finish_clients - start_clients)};

            const auto start_agg{std::chrono::steady_clock::now()};

            std::vector<double> eta(slot_count, LR/(b*number_of_items_in_ctext)); // TODO: Make this b_upt when working with uneven batches
            heongpu::Plaintext<heongpu::Scheme::CKKS> eta_P(context); 
            encoder.encode(eta_P, eta, scale);

            // Generate scaled vectors
            heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_4_scaled;        
            encryptor.encrypt(d_W_4_scaled, all_zeros_plain);
            
            while (eta_P.depth() < d_W_4.depth()) {
                operators.mod_drop_inplace(eta_P);
            }

            operators.multiply_plain_inplace(d_W_4, eta_P);
            operators.rescale_inplace(d_W_4);

            while(d_W_4_scaled.depth() < d_W_4.depth()) {
                operators.mod_drop_inplace(d_W_4_scaled);
            }

            RIS(d_W_4, h[1]*h[2], model_repeat, galois_key_rot, operators, context);
            operators.add_inplace(d_W_4_scaled, d_W_4); 

            while (C_W[3].depth() < d_W_4_scaled.depth()) {
                operators.mod_drop_inplace(C_W[3]);
            }
   
            operators.add_inplace(C_W[3], d_W_4_scaled);

            // std::cout << "C_W before mod drop: " << std::endl;
            // printCiphertext(C_W[3], context, decryptor, encoder);

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_3 before mod drop: " << C_W[3].depth());

            // Gotta do what we goota do so that we can bootstrap
            while (C_W[3].depth() < (number_of_moduli - StoC_piece)) {
                operators.mod_drop_inplace(C_W[3]);
            }

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_3 before bootstrapping: " << C_W[3].depth());
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_b3 = operators.slim_bootstrapping(C_W[3], galois_key_rot, relin_key);

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_b3 after bootstrapping: " << C_W_b3.depth());
            C_W[3] = C_W_b3;

            const auto finish_agg{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_for_agg{(finish_agg - start_agg)};

            total_training_time += elapsed_seconds_for_prep.count() + elapsed_seconds_for_training.count() + elapsed_seconds_for_agg.count();
            // std::cout << "Elapsed time for preparation: " << elapsed_seconds_for_prep.count() << " seconds" << std::endl;
            // std::cout << "Elapsed time for training: " << elapsed_seconds_for_training.count() << " seconds" << std::endl;
            // std::cout << "Elapsed time for aggregation: " << elapsed_seconds_for_agg.count() << " seconds" << std::endl;
            std::cout << "Total elapsed time for this round: " 
                 << elapsed_seconds_for_prep.count() + elapsed_seconds_for_training.count() + elapsed_seconds_for_agg.count() 
                 << " seconds" << std::endl;    

            // std::cout << "C_W after bootstrapping: " << std::endl;
            // printCiphertext(C_W[3], context, decryptor, encoder);
           
        }
    }

    // Decrypt C_W[3] and update the weights
    heongpu::Plaintext<heongpu::Scheme::CKKS> P_W_3(context);
    std::vector<double> W_3;
    decryptor.decrypt(P_W_3, C_W[3]);
    encoder.decode(W_3, P_W_3);
    packed_W[3] = W_3;

    std::cout << "Training finished!" << std::endl;
    std::cout << "Total training time: " << total_training_time << " seconds" << std::endl;

    // After fine-tuning
    EvalResults after = evaluate_model(packed_X_t, y_t, packed_W, chosen_indices,
                                   slot_count, h, m1, m2, m3, m4,
                                   "After Fine-tuning");
    
    std::cout << "Fine-tuning improved subset accuracy by "
          << (after.accuracy_subset - before.accuracy_subset)
          << "%\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        exit (EXIT_FAILURE);
    } catch (...) {
        std::cerr << "An exception occured and program is abruptly terminated" <<std::endl;
        exit (EXIT_FAILURE);
    }    
    return EXIT_SUCCESS;
}
