#include "heongpu.cuh"
#include "../example_util.h"
#include <chrono>

#define DEBUG_DEPTH      0
#define DEBUG_LOG(domain, msg) \
    do { if (domain) std::cout << msg << std::endl; } while(0)

void printCiphertext(heongpu::Ciphertext<heongpu::Scheme::CKKS> C, heongpu::HEContext<heongpu::Scheme::CKKS> context, heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder) {
    heongpu::Plaintext<heongpu::Scheme::CKKS> X_b(context);
    std::vector<double> X_b_vec;
    decryptor.decrypt(X_b, C);
    encoder.decode(X_b_vec, X_b);
    display_vector(X_b_vec, 4UL, 5);
}

std::vector<std::vector<std::vector<double>>> read3DArray(const std::string& filename, int slot_count) {
    std::ifstream file(filename);
    std::vector<std::vector<std::vector<double>>> data3D;
    std::vector<std::vector<double>> current2D;
    std::string line;
    int slot_index = 0;
    std::vector<double> row;

    while (getline(file, line)) {
        if (line.empty()) {
            // blank line means we move on to another client's data
            if (!current2D.empty()) {
                while (slot_index < slot_count){
                    row.push_back(0);
                    slot_index++;
                }
                current2D.push_back(row);
                data3D.push_back(current2D);
                current2D.clear();
                row.clear();
                slot_index = 0;
            }
        } 
        else {
            std::istringstream ss(line);
            double value;
            while (ss >> value) {
                // cout << slot_index << endl;
                if(slot_index == slot_count) {
                    current2D.push_back(row);
                    row.clear();
                    slot_index = 0;
                }
                row.push_back(value);
                slot_index++;
            }
        }
    }
    // push the last 2D slice if any
    if (!current2D.empty()) {
        data3D.push_back(current2D);
    }
    return data3D;
}

std::vector<std::vector<double> > read2DArray(const std::string& filename, int model_repeat, int slot_count) {
    std::ifstream file(filename);
    std::vector<std::vector<double> > data;
    std::string line;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) { 
            row.push_back(value);
        }

        while (int(row.size()) != slot_count) {
            row.push_back(0);
        }
        data.push_back(row);
    }
    return data;
}

std::vector<std::vector<double> > read2DArrayModel(const std::string& filename, int model_repeat) {
    std::ifstream file(filename);
    std::vector<std::vector<double> > data;
    std::string line;
    
    while (getline(file, line)) {
        std::vector<double> row;
        for (int i = 0; i < model_repeat ; i++) {
            std::stringstream ss(line);
            double value;
            while (ss >> value) { 
                row.push_back(value);
            }
        }
        data.push_back(row);
    }
    return data;
}

void RIS(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, int p, int s, heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> copyC(context);
    for (int i = 0; i < log2(s); i++) {
        int shift = p * pow(2, i);
        //  std::cout << "Shift: " << shift << std::endl;
        operators.rotate_rows(C, copyC, galois_key, shift); 
        operators.add_inplace(C, copyC);
       
    }
}

void RR(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C, int p, int s, heongpu::Galoiskey<heongpu::Scheme::CKKS>& galois_key, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context) {
    heongpu::Ciphertext<heongpu::Scheme::CKKS> copyC(context);
    for (int i = 0; i < log2(s); i++) {
        int shift = - p * pow(2, i);
        // std::cout << "Shift: " << shift << std::endl;
        operators.rotate_rows(C, copyC, galois_key, shift); 
        operators.add_inplace(C, copyC);
        
    }
}
// Evaluates the polynomial approximation of the softplus function homomorphically
void Softplus(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS> I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS> &relin_key, double scale, double slot_count, heongpu::HEDecryptor<heongpu::Scheme::CKKS>& decryptor) {   

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
void Sigmoid(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS> I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS>& relin_key, double scale, double slot_count) {

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

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    // Initialize encryption parameters for the CKKS scheme.
    // Initialize encryption parameters for the CKKS scheme.
    heongpu::HEContext<heongpu::Scheme::CKKS> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::sec128);
    size_t poly_modulus_degree = 65536;
    context.set_poly_modulus_degree(poly_modulus_degree);

    // 1761 bits coefficient for 65536
    int number_of_moduli = 30;
    int scale_bits = 50;
    std::vector<int> moduli(number_of_moduli, scale_bits);
    moduli.insert(moduli.begin(), 60);
    context.set_coeff_modulus_bit_sizes(moduli,
       {60, 60, 60}); 
    context.generate();
    context.print_parameters();

    // The scale is set to 2^50, resulting in 50 bits of precision before the
    // decimal point.
    double scale = pow(2.0, scale_bits);

    // Generate keys: the public key for encryption, the secret key for
    // decryption and evaluation key(relinkey) for relinearization.
    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(
        context,
        16); // hamming weight is 16 in this example
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    // Initialize Encoder, Encryptor, Evaluator, and Decryptor. The Encoder will
    // encode the message for SIMD operations. The Encryptor will use the public
    // key to encrypt data, while the Decryptor will use the secret key to
    // decrypt it. The Evaluator will handle operations on the encrypted data.
    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    // heongpu::HEOperator operators(context);
    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(context,
                                                                   encoder);

    // Generate simple vector in CPU.
    const int slot_count = poly_modulus_degree / 2;


    int StoC_piece = 3;
    heongpu::BootstrappingConfig boot_config(3, StoC_piece, 11, false);
    
    operators.generate_bootstrapping_params(scale, boot_config, heongpu::arithmetic_bootstrapping_type::SLIM_BOOTSTRAPPING);

    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key_rot(context, 14);
    keygen.generate_galois_key(galois_key_rot, secret_key,
        heongpu::ExecutionOptions().set_storage_type(heongpu::storage_type::DEVICE));

    std::cout << "Galois key generation finished!" << std::endl;

    // ********************************************************************** //
    // *************************** Start POSEIDON *************************** //
    // ********************************************************************** //

    double N = 10; // number of clients
    double len_X_train = 546;
    double len_X_test = 137;
    // double feature_num = 9; // rounded to the nearest multiple of 2 => 16
    double ell = 2; // number of layers
    std::vector<int> h = {16, 64, 1}; // number of nodes per layer

    double items_per_client = floor(len_X_train / N); // the size of the training set of each client
    double number_of_ctexts_per_client = ceil((items_per_client * h[0] * h[1]) / slot_count); // the number of ciphertexts per client
    double number_of_items_in_ctext = ceil(slot_count / (h[0] * h[1])); // the number of data points that can be packed in a ciphertext

    std::cout << "The # of ciphertexts per client: " << number_of_ctexts_per_client << std::endl;
    std::cout << "The # of items in a ciphertext: " << number_of_items_in_ctext << std::endl;

    // Model parameters
    double round = 50; // the number of federated learning rounds
    int model_repeat = slot_count / (h[0] * h[1]); // the number of times the model is repeated in a ciphertext, equal to number_of_items_in_ctext
    double b = number_of_items_in_ctext / model_repeat; // number of batch ciphertexts, ML batch is 8 and we can store model_repeat of the items in a single ciphertext
    double m = ceil(number_of_ctexts_per_client / b); // the number of batches

    // Read the encoded values
    std::vector<std::vector<std::vector<double>>> packed_X(N, std::vector<std::vector<double>>(number_of_ctexts_per_client, std::vector<double>(slot_count)));
    std::vector<std::vector<double>> packed_X_t(len_X_test, std::vector<double>(slot_count));
    std::vector<std::vector<std::vector<double>>> packed_y(N, std::vector<std::vector<double>>(number_of_ctexts_per_client, std::vector<double>(slot_count)));
    std::vector<std::vector<double>> y_t(len_X_test, std::vector<double>(slot_count)); 
    std::vector<std::vector<double>> packed_W(ell, std::vector<double>(slot_count));

    packed_X = read3DArray("/home/ceren.yildirim/HEonGPU/example/mine/txts_dist/packed_X.txt", slot_count);
    packed_X_t = read2DArray("/home/ceren.yildirim/HEonGPU/example/mine/txts_dist/packed_X_t.txt", model_repeat, slot_count);
    packed_y = read3DArray("/home/ceren.yildirim/HEonGPU/example/mine/txts_dist/packed_y.txt", slot_count);
    y_t = read2DArray("/home/ceren.yildirim/HEonGPU/example/mine/txts_dist/y_test.txt", model_repeat, slot_count);
    packed_W = read2DArrayModel("/home/ceren.yildirim/HEonGPU/example/mine/txts_dist/packed_W.txt", model_repeat);
    // * Don't forget to put newline at the end of the files packed_X, packed_y

    // Generate masks
    std::vector<double> m1(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i % h[0] == 0)
            m1[i] = 1;
    }

    std::vector<double> m2(slot_count, 0);
    for (int i = 0; i < slot_count; i++) {
        if (i%(h[0] * h[1]) < h[2])
            m2[i] = 1;
    }

    // Generate masks for the incomplete batch ciphertexts
    int full_slots = (int(items_per_client) % (int(slot_count) / int(h[0] * h[1])));
    std::vector<double> m1_hlf(slot_count, 0);
    std::cout << "Full slots: " << full_slots << std::endl;
    for (int i = 0; i < full_slots*(h[0]*h[1]); i++) {
        if (i % h[0] == 0)
        m1_hlf[i] = 1;
    }

    std::vector<double> m2_hlf(slot_count, 0);
    for (int i = 0; i < full_slots*(h[0]*h[1]); i++) {
        if (i%(h[0] * h[1]) < h[2])
            m2_hlf[i] = 1;
    }

    // Encode P_W
    std::vector<heongpu::Plaintext<heongpu::Scheme::CKKS>> P_W;
    P_W.reserve(ell);
    for (int i = 0; i < ell; i++) {
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_W_i(context);
            P_W.emplace_back(std::move(P_W_i));
    }
    
    for (int i = 0; i < ell; i++) {
        encoder.encode(P_W[i], packed_W[i], scale);
    }

    std::cout << "Encoding finished!" << std::endl;

    // **** Encrypt the packed values **** //
    // Encrypting the model
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> C_W;
    C_W.reserve(ell);
    for (int i = 0; i < ell; i++) {
        C_W.emplace_back(context);
    }
    for (int i = 0; i < ell; i++) {
        encryptor.encrypt(C_W[i], P_W[i]);
    }
 
    std::cout << "Encryption finished!" << std::endl;

    // Encode softplus coefficients
    double SP_P3 = -0.008601;
    double SP_P2 = 0.111;
    double SP_P1 = 0.5235;
    double SP_P0 = 0.6999;

    // Encode sigmoid coefficients
    double SG_P3 = -0.0005323;
    double SG_P2 = -0.01898;
    double SG_P1 = 0.2048;
    double SG_P0 = 0.5234;

    // **********   TRAINING STARTS HERE   ********** //

    std::cout << "Training starts!" << std::endl;

    double total_training_time = 0.0;

    for (int rn = 0; rn < round; rn++) {
        std::cout << "Round " << rn << std::endl;
        for (int t = 0; t < m; t++) {
            const auto start_prep{std::chrono::steady_clock::now()};

            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> C_W_d11;
            C_W_d11.reserve(ell);
            for (int i = 0; i < ell; i++) {
                C_W_d11.emplace_back(context);
            }

            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1_d14(context);
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2_d12(context);
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1_hlf_d14(context);
            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2_hlf_d12(context);
        
            // Initialize the ciphertexts for the new batch
            std::vector<double> all_zeros(slot_count, 0);
            heongpu::Plaintext<heongpu::Scheme::CKKS> all_zeros_plain(context);
            encoder.encode(all_zeros_plain, all_zeros, scale);
            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> d_W_1;

            for (int i = 0; i < N; i++) {
                heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_1_i(context);
                encryptor.encrypt(d_W_1_i, all_zeros_plain);
                d_W_1.push_back(std::move(d_W_1_i));
            }
            
            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> d_W_2;
            d_W_2.reserve(N);
            for (int i = 0; i < N; i++) {
                d_W_2.emplace_back(context);
            }

            for (int i = 0; i < N; i++) {
                encryptor.encrypt(d_W_2[i], all_zeros_plain);
            }

            const auto start_clients{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_for_prep{(start_clients - start_prep)};

            int b_upt = number_of_items_in_ctext;
            for (int client_id = 0; client_id < N; client_id++) { 
                for (int item = 0; item < b; item++) {
                
                    std::cout << " >>> Round " << rn << " - Client " << client_id << " - Batch " << t << " - Item " << item << std::endl;

                    heongpu::Plaintext<heongpu::Scheme::CKKS> L_0(context);
                    encoder.encode(L_0, packed_X[client_id][t*b+item], scale);
                    heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);

                    while (L_0.depth() < C_W[0].depth()) {
                        operators.mod_drop_inplace(L_0); 
                    }
                    
                    operators.multiply_plain(C_W[0], L_0, U_1); // U_1 Depth 1
                    operators.rescale_inplace(U_1);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_1 after multiply: " << U_1.depth());

                    RIS(U_1, 1, h[0], galois_key_rot, operators, context);

                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1;
                    encoder.encode(P_m1, m1, scale);
                    
                    while (P_m1.depth() < U_1.depth()) {
                        operators.mod_drop_inplace(P_m1); 
                    }
                    
                    operators.multiply_plain_inplace(U_1, P_m1); // U_1 Depth 2
                    operators.rescale_inplace(U_1);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth of U_1 after RIS: " << U_1.depth());
                    RR(U_1, 1, h[2], galois_key_rot, operators, context);

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);
                    Softplus(L_1, U_1, SP_P3, SP_P2, SP_P1, SP_P0, operators, context, encoder, relin_key, scale, slot_count, decryptor);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth L_1 after softplus: " << L_1.depth());

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);

            
                    while (L_1.depth() > C_W[1].depth()) {
                        operators.mod_drop_inplace(C_W[1]); 
                    }
                
                    while (C_W[1].depth() > L_1.depth()) {
                        operators.mod_drop_inplace(L_1); 
                    }
                
                    operators.multiply(L_1, C_W[1], U_2);
                    operators.relinearize_inplace(U_2, relin_key);
                    operators.rescale_inplace(U_2);
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after multiply: " << U_2.depth());

                    RIS(U_2, h[0], h[1], galois_key_rot, operators, context);
                    
                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
                    encoder.encode(P_m2, m2, scale);

                    while (U_2.depth() > P_m2.depth()) {
                        operators.mod_drop_inplace(P_m2); 
                    }
                
                    operators.multiply_plain_inplace(U_2, P_m2); 
                    encoder.encode(P_m2, m2, scale); // why bootstrap when you can just...
                    operators.rescale_inplace(U_2);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after RIS: " << U_2.depth());                      

                    // ! Bootstrap
                    while (U_2.depth() < number_of_moduli - StoC_piece) {
                        operators.mod_drop_inplace(U_2);
                    
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 before bootstrapping: " << U_2.depth());
                    U_2 = operators.slim_bootstrapping(U_2, galois_key_rot, relin_key);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after bootstrapping: " << U_2.depth());

                    // // Decrypt and re-encrypt U_2 ciphertexts
                    // heongpu::Plaintext<heongpu::Scheme::CKKS> p_U_2(context);
                    // decryptor.decrypt(p_U_2, U_2);
                    // // decode
                    // std::vector<double> U_2_vec(slot_count, 0);
                    // encoder.decode(U_2_vec, p_U_2);

                    // // encode
                    // encoder.encode(p_U_2, U_2_vec, scale);
                    // heongpu::Ciphertext<heongpu::Scheme::CKKS> c_U_2(context);
                    // encryptor.encrypt(c_U_2, p_U_2);

                    // U_2 = c_U_2; // Re-encrypt E_1 ciphertext

                    // while (U_2.depth() < 21) {
                    //     operators.mod_drop_inplace(U_2);
                    // }


                    heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
                    Sigmoid(L_2, U_2, SG_P3, SG_P2, SG_P1, SG_P0, operators, context, encoder, relin_key, scale, slot_count);
                    // L_2 Depth 10 / 23
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth L_2 after sigmoid: " << L_2.depth());
            
                    // ****************** Backpropagation ****************** //
                    DEBUG_LOG(DEBUG_DEPTH, " >> Backpropagation starts!");

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> E_2(context);
                    operators.negate(L_2, E_2); 
                    heongpu::Plaintext<heongpu::Scheme::CKKS> P_y_curr(context);
                    encoder.encode(P_y_curr, packed_y[client_id][t*b+item], scale);

                    while (P_y_curr.depth() < E_2.depth()) {
                        operators.mod_drop_inplace(P_y_curr);
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth P_y_curr after mod drop: " << P_y_curr.depth());
                    
                    operators.add_plain_inplace(E_2, P_y_curr);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 after add: " << E_2.depth() << " - should be 23");
                    heongpu::Ciphertext<heongpu::Scheme::CKKS> d(context);
                    
                    if (item == b-1 && t == m-1) {
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2_hlf;
                        encoder.encode(P_m2_hlf, m2_hlf, scale);
                        while (P_m2_hlf.depth() < E_2.depth()) {
                            operators.mod_drop_inplace(P_m2_hlf); 
                        }
                        
                        operators.multiply_plain_inplace(E_2, P_m2_hlf); 
                        operators.rescale_inplace(E_2);
                    }
                    else
                    {
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
                        encoder.encode(P_m2, m2, scale);
                        while (P_m2.depth() < E_2.depth()) {
                                operators.mod_drop_inplace(P_m2); 
                        }
                        operators.multiply_plain_inplace(E_2, P_m2); 
                        operators.rescale_inplace(E_2);
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 after multiply: " << E_2.depth() << " - should be 24");

                    RR(E_2, h[0], h[1], galois_key_rot, operators, context);

                    heongpu::Ciphertext<heongpu::Scheme::CKKS> interm(context);

                    while (L_1.depth() < E_2.depth()) {
                        operators.mod_drop_inplace(L_1); 
                    }
                    while (E_2.depth() < L_1.depth()) {
                        operators.mod_drop_inplace(E_2); 
                    }

                    operators.multiply(E_2, L_1, interm); 
                    operators.relinearize_inplace(interm, relin_key);
                    operators.rescale_inplace(interm);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth of interm after multiply: " << interm.depth() << " - should be 26");

                    while (d_W_2[client_id].depth() < interm.depth()) {
                        operators.mod_drop_inplace(d_W_2[client_id]);
                    }

                    operators.add_inplace(d_W_2[client_id], interm); 
                    heongpu::Ciphertext<heongpu::Scheme::CKKS> E_1(context);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_d11[1] before mod drop: " << C_W_d11[1].depth());
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 before mod drop: " << E_2.depth());
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W[1] before mod drop: " << C_W[1].depth());
    
                    int iter = 0;
                    while (C_W_d11[1].depth() < E_2.depth()) {
                        if (iter == 0) {
                            C_W_d11[1] = C_W[1];
                            iter++;
                        }
                        else
                            operators.mod_drop_inplace(C_W_d11[1]);
                    }

                    while (C_W_d11[1].depth() > E_2.depth()) {
                        operators.mod_drop_inplace(E_2); 
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_d11[1] after mod drop: " << C_W_d11[1].depth());
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 after mod drop: " << E_2.depth());

                    operators.multiply(E_2, C_W_d11[1], E_1); 
                    operators.relinearize_inplace(E_1, relin_key);
                    operators.rescale_inplace(E_1);

                    RIS(E_1, 1, h[2], galois_key_rot, operators, context);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 after RIS: " << E_1.depth() << " - should be 26");
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth U_1 before sigmoid: " << U_1.depth());
                    Sigmoid(d, U_1, SG_P3, SG_P2, SG_P1, SG_P0, operators, context, encoder, relin_key, scale, slot_count);
                    
                    while (d.depth() < E_1.depth()) {
                        operators.mod_drop_inplace(d); 
                    }
        
                    while (E_1.depth() > d.depth()) {
                        operators.mod_drop_inplace(E_1);
                    }
                    
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth d after sigmoid: " << d.depth());
                    operators.multiply_inplace(E_1, d);
                    operators.relinearize_inplace(E_1, relin_key);
                    operators.rescale_inplace(E_1);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 after multiply: " << E_1.depth());

                    // ! Bootstrap
                    while (E_1.depth() < number_of_moduli - StoC_piece) {
                        operators.mod_drop_inplace(E_1);
                    
                    }

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 before bootstrapping: " << E_1.depth());
                    E_1 = operators.slim_bootstrapping(E_1, galois_key_rot, relin_key);
                    DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 after bootstrapping: " << E_1.depth());

                    // // Decrypt and re-encrypt E_1 ciphertexts
                    // heongpu::Plaintext<heongpu::Scheme::CKKS> p_E_1(context);
                    // decryptor.decrypt(p_E_1, E_1);
                    // // decode
                    // std::vector<double> E_1_vec(slot_count, 0);
                    // encoder.decode(E_1_vec, p_E_1);

                    // // encode
                    // encoder.encode(p_E_1, E_1_vec, scale);
                    // heongpu::Ciphertext<heongpu::Scheme::CKKS> c_E_1(context);
                    // encryptor.encrypt(c_E_1, p_E_1);

                    // E_1 = c_E_1; // Re-encrypt E_1 ciphertext
            
                    // while (E_1.depth() < 21) {
                    //     operators.mod_drop_inplace(E_1);
                    // }

                    
                    if (item == b-1 && t == m-1) {
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1_hlf(context);
                        encoder.encode(P_m1_hlf, m1_hlf, scale);
                        while(P_m1_hlf.depth() < E_1.depth()) {
                                operators.mod_drop_inplace(P_m1_hlf);
                        }
                        operators.multiply_plain_inplace(E_1, P_m1_hlf); 
                        operators.rescale_inplace(E_1);
                    }
                    else {
                        encoder.encode(P_m1, m1, scale);
                        while(P_m1.depth() < E_1.depth()) {
                            operators.mod_drop_inplace(P_m1); 
                        }
                        operators.multiply_plain_inplace(E_1, P_m1); 
                        operators.rescale_inplace(E_1);
                    }

                    // Check the depth of P_m1 and P_m1_d14 
                    // DEBUG_LOG(DEBUG_DEPTH, " !!! Depth P_m1_d14 after mod drop: " << P_m1_d14.depth());
                    // DEBUG_LOG(DEBUG_DEPTH, " !!! Depth P_m1 after mod drop: " << P_m1.depth());
                    // DEBUG_LOG(DEBUG_DEPTH, " !!! Depth E_1 after multiply: " << E_1.depth());
                    
                    RR(E_1, 1, h[0], galois_key_rot, operators, context);
                    
                    operators.mod_drop_inplace(L_0); 
                    while (L_0.depth() < E_1.depth()) {
                        operators.mod_drop_inplace(L_0); 
                    }

                    operators.multiply_plain(E_1, L_0, interm); 
                    operators.rescale_inplace(interm);

                    DEBUG_LOG(DEBUG_DEPTH, " > Depth after multiply: " << interm.depth());

                    while (d_W_1[client_id].depth() < interm.depth()) {
                        operators.mod_drop_inplace(d_W_1[client_id]); 
                    }
                    
                    operators.add_inplace(d_W_1[client_id], interm);
                    // printCiphertext(d_W_1[client_id], context, decryptor, encoder);
                    DEBUG_LOG(DEBUG_DEPTH, "\n");
                    if (item == b-1 && t == m-1) {
                        b_upt = full_slots;
                    }
                }
            }

            const auto finish_clients{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_per_client{(finish_clients - start_clients)/ N};

            std::cout << "Elapsed time for each client: " << elapsed_seconds_per_client.count() << " seconds." << std::endl;

            const auto start_agg{std::chrono::steady_clock::now()};

            std::vector<double> eta(slot_count, 0.1/(b_upt*N)); 
            heongpu::Plaintext<heongpu::Scheme::CKKS> eta_P_1(context); heongpu::Plaintext<heongpu::Scheme::CKKS> eta_P_2(context);
            encoder.encode(eta_P_1, eta, scale); encoder.encode(eta_P_2, eta, scale);

            // Generate scaled vectors
            heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_1_scaled;        
            encryptor.encrypt(d_W_1_scaled, all_zeros_plain);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_2_scaled;
            encryptor.encrypt(d_W_2_scaled, all_zeros_plain);

            // Each client holds the dW value in pieces because of batching, sum them up and scale
            for (int j = 0; j < N; j++) {

                while (eta_P_1.depth() < d_W_1[j].depth()) {
                    operators.mod_drop_inplace(eta_P_1);
                }

                operators.multiply_plain_inplace(d_W_1[j], eta_P_1);
                operators.rescale_inplace(d_W_1[j]);

                while(d_W_1_scaled.depth() < d_W_1[j].depth()) {
                    operators.mod_drop_inplace(d_W_1_scaled);
                }

                RIS(d_W_1[j], h[0]*h[1], model_repeat, galois_key_rot, operators, context);
                operators.add_inplace(d_W_1_scaled, d_W_1[j]); 

                while (eta_P_2.depth() < d_W_2[j].depth()) {
                    operators.mod_drop_inplace(eta_P_2);
                } 
                operators.multiply_plain_inplace(d_W_2[j], eta_P_2); 
                operators.rescale_inplace(d_W_2[j]);
                
                while(d_W_2_scaled.depth() < d_W_2[j].depth()) {
                    operators.mod_drop_inplace(d_W_2_scaled);
                }
                RIS(d_W_2[j], h[0]*h[1], model_repeat, galois_key_rot, operators, context); // !
                operators.add_inplace(d_W_2_scaled, d_W_2[j]);
            }

            printCiphertext(d_W_1_scaled, context, decryptor, encoder);

            while (C_W[0].depth() < d_W_1_scaled.depth()) {
                operators.mod_drop_inplace(C_W[0]);
            }
            while (C_W[1].depth() < d_W_2_scaled.depth()) {
                operators.mod_drop_inplace(C_W[1]);
            }

            operators.add_inplace(C_W[0], d_W_1_scaled);
            operators.add_inplace(C_W[1], d_W_2_scaled);

            // std::cout << "C_W before mod drop: " << std::endl;
            // printCiphertext(C_W[0], context, decryptor, encoder);

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_0 before mod drop: " << C_W[0].depth());
            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_1 before mod drop: " << C_W[1].depth());

            // Gotta do what we goota do so that we can bootstrap
            while (C_W[0].depth() < (number_of_moduli - StoC_piece)) {
                operators.mod_drop_inplace(C_W[0]);
            }

            while (C_W[1].depth() < (number_of_moduli - StoC_piece)) {
                operators.mod_drop_inplace(C_W[1]);
            }

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_0 before bootstrapping: " << C_W[0].depth());
            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_1 before bootstrapping: " << C_W[1].depth());
            // printCiphertext(C_W[0], context, decryptor, encoder);
            // Bootstrap the C_W ciphertexts and make them ready for the next round

            // std::cout << "Bootstrapping starts!" << std::endl;
        
            heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_b0 = operators.slim_bootstrapping(C_W[0], galois_key_rot, relin_key);
            heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_b1 = operators.slim_bootstrapping(C_W[1], galois_key_rot, relin_key);
            // std::cout << "Bootstrapping finished!" << std::endl;
            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_b0 after bootstrapping: " << C_W_b0.depth());
            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_b1 after bootstrapping: " << C_W_b1.depth());

            // // Decrypt and re-encrypt C_W ciphertexts
            // heongpu::Plaintext<heongpu::Scheme::CKKS> p_C_W_b0(context);
            // heongpu::Plaintext<heongpu::Scheme::CKKS> p_C_W_b1(context);
            // decryptor.decrypt(p_C_W_b0, C_W[0]);
            // decryptor.decrypt(p_C_W_b1, C_W[1]);    
            // // decode
            // std::vector<double> C_W_0_vec(slot_count, 0);
            // encoder.decode(C_W_0_vec, p_C_W_b0);
            // std::vector<double> C_W_1_vec(slot_count, 0);
            // encoder.decode(C_W_1_vec, p_C_W_b1);
            // // encode
            // encoder.encode(p_C_W_b0, C_W_0_vec, scale);
            // encoder.encode(p_C_W_b1, C_W_1_vec, scale);
            // heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_b0(context);
            // heongpu::Ciphertext<heongpu::Scheme::CKKS> C_W_b1(context);
            // encryptor.encrypt(C_W_b0, p_C_W_b0);
            // encryptor.encrypt(C_W_b1, p_C_W_b1);    


            C_W[0] = C_W_b0;
            C_W[1] = C_W_b1;


            const auto finish_agg{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_for_agg{(finish_agg - start_agg)};

            total_training_time += elapsed_seconds_for_prep.count() + elapsed_seconds_per_client.count() + elapsed_seconds_for_agg.count();
            std::cout << "Elapsed time for preparation: " << elapsed_seconds_for_prep.count() << " seconds" << std::endl;
            std::cout << "Elapsed time for clients: " << elapsed_seconds_per_client.count() << " seconds" << std::endl;
            std::cout << "Elapsed time for aggregation: " << elapsed_seconds_for_agg.count() << " seconds" << std::endl;
            std::cout << "Total elapsed time for this round: " 
                 << (elapsed_seconds_for_prep.count() + elapsed_seconds_per_client.count() + elapsed_seconds_for_agg.count()) 
                 << " seconds" << std::endl;    

            std::cout << "C_W aftrer bootstrapping: " << std::endl;
            printCiphertext(C_W[0], context, decryptor, encoder);
            
        }
    }

    std::cout << "Training finished!" << std::endl;
    std::cout << "Total training time: " << total_training_time << " seconds" << std::endl;

    // **********   TRAINING ENDS HERE   ********** //

    // Decrypt and output the C_W ciphertexts to a txt file
    heongpu::Plaintext<heongpu::Scheme::CKKS> p_C_W_0(context);
    heongpu::Plaintext<heongpu::Scheme::CKKS> p_C_W_1(context);
    decryptor.decrypt(p_C_W_0, C_W[0]);
    decryptor.decrypt(p_C_W_1, C_W[1]);
    // decode
    std::vector<double> C_W_0_vec(slot_count, 0);
    std::vector<double> C_W_1_vec(slot_count, 0);
    encoder.decode(C_W_0_vec, p_C_W_0);
    encoder.decode(C_W_1_vec, p_C_W_1); 
    // write to file
    std::ofstream out_C_W_0("C_W_0.txt");
    std::ofstream out_C_W_1("C_W_1.txt");
    if (out_C_W_0.is_open() && out_C_W_1.is_open()) {
        for (const auto& val : C_W_0_vec) {
            out_C_W_0 << val << std::endl;
        }
        for (const auto& val : C_W_1_vec) {
            out_C_W_1 << val << std::endl;
        }
        out_C_W_0.close();      
        out_C_W_1.close();
        std::cout << "C_W ciphertexts written to files C_W_0.txt and C_W_1.txt" << std::endl;
    } else {
        std::cerr << "Unable to open files for writing C_W ciphertexts." << std::endl;
        return EXIT_FAILURE;
    }

    // **********   TESTING STARTS HERE   ********** //

    std::cout << "Testing starts!" << std::endl;
    
    int corr = 0;
    for (int t = 0; t < packed_X_t.size(); t++) {
        heongpu::Plaintext<heongpu::Scheme::CKKS> L_0(context); 
        encoder.encode(L_0, packed_X_t[t], scale);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);

        while (L_0.depth() < C_W[0].depth()) {
            operators.mod_drop_inplace(L_0); 
        }
        
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
        DEBUG_LOG(DEBUG_DEPTH, "Depth after RIS: " << U_1.depth());

        RR(U_1, 1, h[2], galois_key_rot, operators, context);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);
        Softplus(L_1, U_1, SP_P3, SP_P2, SP_P1, SP_P0, operators, context, encoder, relin_key, scale, slot_count, decryptor);
        DEBUG_LOG(DEBUG_DEPTH, " > Depth L_1 after softplus: " << L_1.depth());

        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);

        while (L_1.depth() > C_W[1].depth()) {
            operators.mod_drop_inplace(C_W[1]); 
        }

        while (C_W[1].depth() > L_1.depth()) {
            operators.mod_drop_inplace(L_1); 
        }
       
        operators.multiply(L_1, C_W[1], U_2); 
        operators.relinearize_inplace(U_2, relin_key);
        operators.rescale_inplace(U_2);
        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after multiply: " << U_2.depth());

        RIS(U_2, h[0], h[1], galois_key_rot, operators, context);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
        encoder.encode(P_m2, m2, scale);       


        while (U_2.depth() > P_m2.depth()) {
            operators.mod_drop_inplace(P_m2); 
        }

        while (P_m2.depth() > U_2.depth()) {
            operators.mod_drop_inplace(U_2); 
        }
 
        operators.multiply_plain_inplace(U_2, P_m2);
        operators.rescale_inplace(U_2);
        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after RIS: " << U_2.depth());

        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
        Sigmoid(L_2, U_2, SG_P3, SG_P2, SG_P1, SG_P0, operators, context, encoder, relin_key, scale, slot_count);

        DEBUG_LOG(DEBUG_DEPTH, " > Depth L_2 after sigmoid: " << L_2.depth());

        heongpu::Plaintext<heongpu::Scheme::CKKS> L_2_decrypted(context);
        std::vector<double> L_2_decrypted_vec(slot_count, 0);
        decryptor.decrypt(L_2_decrypted, L_2);
        encoder.decode(L_2_decrypted_vec, L_2_decrypted);

        int y = 1;

        if (L_2_decrypted_vec[0] >= 0.5)
            y = 1;
        else
            y = 0;

        if (int(y_t[t][0]) == 1 && y == 1)
            corr += 1;
        else if  (int(y_t[t][0]) == 0 && y == 0)
            corr += 1;
    }

    std::cout << "Accuracy is: " << (corr / len_X_test) * 100 << std::endl;   
    std::cout << corr << " correct predictions out of " << len_X_test << " samples." << std::endl; 

    return EXIT_SUCCESS;
}