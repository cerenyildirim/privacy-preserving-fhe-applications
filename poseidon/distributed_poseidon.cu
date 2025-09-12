#include "heongpu.cuh"
#include "../example_util.h"
#include <omp.h>
#include <chrono>

#define DEBUG_DEPTH      1
#define DEBUG_LOG(domain, msg) \
    do { if (domain) std::cout << msg << std::endl; } while(0)

void printCiphertext(heongpu::Ciphertext<heongpu::Scheme::CKKS> C, heongpu::HEContext<heongpu::Scheme::CKKS> context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, std::vector<heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS>> &mpc_managers, std::vector<heongpu::Secretkey<heongpu::Scheme::CKKS>> &secret_keys, int N) {

    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> partial_ciphertexts;
    // Generate the ciphertexts
    for (int i = 0; i < N; i++) {
        heongpu::Ciphertext<heongpu::Scheme::CKKS> partial_ciphertext(context);
        partial_ciphertexts.push_back(std::move(partial_ciphertext));
    }

    for (int i = 0; i < N; i++) {
        mpc_managers[i].decrypt_partial(
            C, secret_keys[i], partial_ciphertexts[i]); 
    }

    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext_result(context);
    mpc_managers[0].decrypt(partial_ciphertexts, plaintext_result);

    std::vector<double> check_result;
    encoder.decode(check_result, plaintext_result);

    display_vector(check_result);
}

void collectiveBootstrapping(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C,
                             std::vector<heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS>>& mpc_managers,
                             heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS>& mpc_manager_server,
                             std::vector<heongpu::Secretkey<heongpu::Scheme::CKKS>>& secret_keys,
                             heongpu::HEContext<heongpu::Scheme::CKKS>& context,
                             heongpu::RNGSeed& common_seed,
                             int N) {

    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> all_boot_ciphertexts;

    for (int i = 0; i < N; i++) {
        heongpu::Ciphertext<heongpu::Scheme::CKKS> boot_ciphertext(context);
        all_boot_ciphertexts.push_back(std::move(boot_ciphertext));
        mpc_managers[i].distributed_bootstrapping_participant(
            C, all_boot_ciphertexts[i], secret_keys[i], common_seed);
    }

    heongpu::Ciphertext<heongpu::Scheme::CKKS> boot_server_ciphertext(context);
    mpc_manager_server.distributed_bootstrapping_coordinator(
        all_boot_ciphertexts, C, boot_server_ciphertext,
        common_seed);

    C = boot_server_ciphertext; 

}

std::vector<double> collectiveDecryption(heongpu::Ciphertext<heongpu::Scheme::CKKS>& C,
                                        std::vector<heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS>>& mpc_managers,
                                        std::vector<heongpu::HEEncoder<heongpu::Scheme::CKKS>>& encoders,
                                        std::vector<heongpu::Secretkey<heongpu::Scheme::CKKS>>& secret_keys,
                                        heongpu::HEContext<heongpu::Scheme::CKKS>& context,
                                        int N) {

    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> partial_ciphertexts;
    for (int i = 0; i < N; i++) {
        heongpu::Ciphertext<heongpu::Scheme::CKKS> partial_ciphertext(context);
        partial_ciphertexts.push_back(std::move(partial_ciphertext));

        mpc_managers[i].decrypt_partial(
            C, secret_keys[i],
            partial_ciphertexts[i]); 
        }

    heongpu::Plaintext<heongpu::Scheme::CKKS> plaintext_result(context);
    mpc_managers[0].decrypt(partial_ciphertexts, plaintext_result);

    std::vector<double> decrypted_vec;
    encoders[0].decode(decrypted_vec, plaintext_result);
    return decrypted_vec;
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
void Softplus(heongpu::Ciphertext<heongpu::Scheme::CKKS>& R, heongpu::Ciphertext<heongpu::Scheme::CKKS> I, double PA3, double PA2, double PA1, double PA0, heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>& operators, heongpu::HEContext<heongpu::Scheme::CKKS>& context, heongpu::HEEncoder<heongpu::Scheme::CKKS>& encoder, heongpu::Relinkey<heongpu::Scheme::CKKS> &relin_key, double scale, double slot_count) {   

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


int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    heongpu::HEContext<heongpu::Scheme::CKKS> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);

    size_t poly_modulus_degree = 16384; 
    const int slot_count = poly_modulus_degree / 2;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_bit_sizes({60, 50, 50, 50, 50, 50, 50, 50}, {60}); // 16384: 438
    context.generate();
    context.print_parameters();

    double scale = pow(2.0, 50);
    int depth_val = 6;

    double N = 10; // number of clients

    heongpu::RNGSeed common_seed; // automatically generate itself

    std::vector<int> shift_value = {0, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512};
    for (int i = 0; i < log2(slot_count); i++) {
        shift_value.push_back(pow(2, i));
    }

    std::vector<heongpu::Secretkey<heongpu::Scheme::CKKS>> secret_keys;
    std::vector<heongpu::HEEncoder<heongpu::Scheme::CKKS>> encoders;

    std::vector<heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS>> mpc_managers;
    std::vector<heongpu::MultipartyPublickey<heongpu::Scheme::CKKS>> participant_public_keys;
    std::vector<heongpu::MultipartyRelinkey<heongpu::Scheme::CKKS>> participant_relin_keys_stage1;
    std::vector<heongpu::MultipartyGaloiskey<heongpu::Scheme::CKKS>> participant_galois_keys;

    for (int i = 0; i < N; i++) {
        heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
        heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(context);
        keygen.generate_secret_key(secret_key);
        secret_keys.push_back(std::move(secret_key));

        heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
        encoders.push_back(std::move(encoder));

        heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS> mpc_manager(
            context, encoders[i], scale);
        mpc_managers.push_back(std::move(mpc_manager));

        // Publickey
        heongpu::MultipartyPublickey<heongpu::Scheme::CKKS> public_key(
            context, common_seed);
        mpc_managers[i].generate_public_key_share(public_key, secret_keys[i]);
        participant_public_keys.push_back(std::move(public_key));

        // Relinkey
        heongpu::MultipartyRelinkey<heongpu::Scheme::CKKS> relin_key_stage1(
            context, common_seed);
        mpc_managers[i].generate_relin_key_init(relin_key_stage1, secret_keys[i]);
        participant_relin_keys_stage1.push_back(std::move(relin_key_stage1));

        // Galoiskey
        heongpu::MultipartyGaloiskey<heongpu::Scheme::CKKS> galois_key(
            context, shift_value, common_seed);
        mpc_managers[i].generate_galois_key_share(galois_key, secret_keys[i]);
        participant_galois_keys.push_back(std::move(galois_key));

    }


    ///////////////////////////////////////////////////////////
    ///////////// Key Sharing (Stage 1) (Phases 1) ////////////
    ///////////////////////////////////////////////////////////

    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder_server(context);
    heongpu::HEMultiPartyManager<heongpu::Scheme::CKKS> mpc_manager_server(
        context, encoders[0], scale);

    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen_server(context);
    heongpu::Publickey<heongpu::Scheme::CKKS> common_public_key(context);
    mpc_manager_server.assemble_public_key_share(participant_public_keys,
                                                 common_public_key);

    heongpu::MultipartyRelinkey<heongpu::Scheme::CKKS> common_relin_key_stage1(
        context, common_seed);
    mpc_manager_server.assemble_relin_key_init(participant_relin_keys_stage1,
                                               common_relin_key_stage1);

    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context, shift_value);
    mpc_manager_server.assemble_galois_key_share(participant_galois_keys,
                                                 galois_key);

    std::cout << "Key sharing (Stage 1) completed successfully!" << std::endl;

    ///////////////////////////////////////////////////////////
    ///////////// Setup (Stage 1) (Phases 2) ////////////
    ///////////////////////////////////////////////////////////

    std::vector<heongpu::MultipartyRelinkey<heongpu::Scheme::CKKS>> participant_relin_keys_stage2;

    for (int i = 0; i < N; i++) {
        heongpu::MultipartyRelinkey<heongpu::Scheme::CKKS> relin_key_stage2(
        context, common_seed);
        participant_relin_keys_stage2.push_back(relin_key_stage2);

        mpc_managers[i].generate_relin_key_share(
            common_relin_key_stage1, participant_relin_keys_stage2[i], secret_keys[i]);

    }

    std::cout << "Setup (Stage 1) completed successfully!" << std::endl;
    ///////////////////////////////////////////////////////////
    //////////// Key Sharing (Stage 1) (Phases 2) /////////////
    ///////////////////////////////////////////////////////////

    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    mpc_manager_server.assemble_relin_key_share(participant_relin_keys_stage2,
                                                common_relin_key_stage1,
                                                relin_key);


    heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS> operators(
        context, encoders[0]);

    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor_alice(
        context, common_public_key);
 
    // ********************************************************************** //
    // *************************** Start POSEIDON *********ß****************** //
    // ********************************************************************** //

    
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
    double round = 15; // the number of federated learning rounds
    int model_repeat = slot_count / (h[0] * h[1]); // the number of times the model is repeated in a ciphertext, equal to number_of_items_in_ctext
    double b = number_of_items_in_ctext / model_repeat; // number of batch ciphertexts, ML batch is 8 and we can store model_repeat of the items in a single ciphertext
    double m = ceil(number_of_ctexts_per_client / b); // the number of batches

    std::cout << "b is: " << b << std::endl;
    std::cout << "m is: " << m << std::endl;

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

    std::cout << "Files read successfully!" << std::endl;

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
    for (int i = 0; i < full_slots*(h[0]*h[1]); i++) {
        if (i % h[0] == 0)
        m1_hlf[i] = 1;
    }

    std::vector<double> m2_hlf(slot_count, 0);
    for (int i = 0; i < full_slots*(h[0]*h[1]); i++) {
        if (i%(h[0] * h[1]) < h[2])
            m2_hlf[i] = 1;
    }

    std::cout << "Masks generated successfully!" << std::endl;

    // **** Encode the packed values **** //

    // Encode P_W
    std::vector<heongpu::Plaintext<heongpu::Scheme::CKKS> > P_W;
    P_W.reserve(ell);
    for (int i = 0; i < ell; i++) {
            heongpu::Plaintext<heongpu::Scheme::CKKS>  P_W_i(context);
            P_W.emplace_back(std::move(P_W_i));
    }
    
    for (int i = 0; i < ell; i++) {
        encoders[0].encode(P_W[i], packed_W[i], scale);
    }

    std::cout << "P_W encoded successfully!" << std::endl;

    // Encode softplus coefficients
    double SP_C3 = -0.008601;
    double SP_C2 = 0.111;
    double SP_C1 = 0.5235;
    double SP_C0 = 0.6999;

    // Encode sigmoid coefficients
    double SG_C3 = -0.0005323;
    double SG_C2 = -0.01898;
    double SG_C1 = 0.2048;
    double SG_C0 = 0.5234;

    std::cout << "Encoding finished!" << std::endl;

    // **** Encrypt the packed values **** //
    // Encrypting the model
    std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> C_W;
    C_W.reserve(ell);
    for (int i = 0; i < ell; i++) {
        C_W.emplace_back(context);
    }
    for (int i = 0; i < ell; i++) {
        encryptor_alice.encrypt(C_W[i], P_W[i]);
    }
 
    std::cout << "Encryption finished!" << std::endl;

    // **********   TRAINING STARTS HERE   ********** //

    std::cout << "Training starts!" << std::endl;

    double total_training_time = 0.0;

    for (int rn = 0; rn < round; rn++) {
        std::cout << "Round " << rn << std::endl;
        for (int t = 0; t < m; t++) {
            const auto start_prep{std::chrono::steady_clock::now()};

            heongpu::Plaintext<heongpu::Scheme::CKKS>  P_m1_d14(context);
            heongpu::Plaintext<heongpu::Scheme::CKKS>  P_m2_d12(context);
            heongpu::Plaintext<heongpu::Scheme::CKKS>  P_m1_hlf_d14(context);
            heongpu::Plaintext<heongpu::Scheme::CKKS>  P_m2_hlf_d12(context);
        
            // Initialize the ciphertexts for the new batch
            std::vector<double> all_zeros(slot_count, 0);
            heongpu::Plaintext<heongpu::Scheme::CKKS>  all_zeros_plain(context);
            encoders[0].encode(all_zeros_plain, all_zeros, scale);
            
            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> d_W_1;
            std::vector<heongpu::Ciphertext<heongpu::Scheme::CKKS>> d_W_2;

            const auto end_singular_prep{std::chrono::steady_clock::now()};
  
            for (int i = 0; i < N; i++) {
                heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_1_i(context);
                encryptor_alice.encrypt(d_W_1_i, all_zeros_plain);
                d_W_1.push_back(std::move(d_W_1_i));
                heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_2_i(context);
                encryptor_alice.encrypt(d_W_2_i, all_zeros_plain);
                d_W_2.push_back(std::move(d_W_2_i));
            }

            const auto start_clients{std::chrono::steady_clock::now()};

            const std::chrono::duration<double> elapsed_seconds_for_prep{(end_singular_prep - start_prep + (start_clients - end_singular_prep)/N)};
            
            for (int client_id = 0; client_id < N; client_id++) {
                for (int item = 0; item < b; item++) {
                    if (t*b+item < number_of_ctexts_per_client) {
                        std::cout << " >>> Round " << rn << " - Client " << client_id << " - Batch " << t << " - Item " << item << std::endl;

                        heongpu::Plaintext<heongpu::Scheme::CKKS>  L_0(context);
                        encoders[0].encode(L_0, packed_X[client_id][t*b+item], scale); // L_0 Depth 1
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);

                        while (L_0.depth() < C_W[0].depth()) {
                            operators.mod_drop_inplace(L_0); 
                        }
                        
                        operators.multiply_plain(C_W[0], L_0, U_1); // U_1 Depth 1
                        operators.rescale_inplace(U_1);

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_1 after multiply: " << U_1.depth());

                        RIS(U_1, 1, h[0], galois_key, operators, context);
                        
                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1;
                        encoders[0].encode(P_m1, m1, scale);
                        
                        while (P_m1.depth() < U_1.depth()) {
                            operators.mod_drop_inplace(P_m1); 
                        }
                       
                        operators.multiply_plain_inplace(U_1, P_m1); // U_1 Depth 2
                        operators.rescale_inplace(U_1);
                        DEBUG_LOG(DEBUG_DEPTH, "Depth U_1 after RIS: " << U_1.depth());
                        RR(U_1, 1, h[2], galois_key, operators, context);

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);
                        Softplus(L_1, U_1, SP_C3, SP_C2, SP_C1, SP_C0, operators, context, encoders[0], relin_key, scale, slot_count);
                        // L_1 Depth 5
                        DEBUG_LOG(DEBUG_DEPTH, " > Depth L_1 after softplus: " << L_1.depth());

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);

                        if (L_1.depth() > C_W[1].depth()) {
                            while (L_1.depth() > C_W[1].depth()) {
                                operators.mod_drop_inplace(C_W[1]); // Ensure L_1 matches C_W[1] Depth
                            }
                        
                        }
                       
                        operators.multiply(L_1, C_W[1], U_2); // U_2 Depth 6
                        operators.relinearize_inplace(U_2, relin_key);
                        operators.rescale_inplace(U_2);
                        
                        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after multiply: " << U_2.depth());

                        RIS(U_2, h[0], h[1], galois_key, operators, context);

                        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
                        encoders[0].encode(P_m2, m2, scale);
                        
                        while (U_2.depth() > P_m2.depth()) {
                            operators.mod_drop_inplace(P_m2); 
                        }
                    
                        operators.multiply_plain_inplace(U_2, P_m2); // U_2 Depth 7
                        encoders[0].encode(P_m2, m2, scale); // why bootstrap when you can just...
                        operators.rescale_inplace(U_2);
                        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after RIS: " << U_2.depth());                      

                        collectiveBootstrapping(U_2, mpc_managers, mpc_manager_server,
                            secret_keys, context, common_seed, N);

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
                        Sigmoid(L_2, U_2, SG_C3, SG_C2, SG_C1, SG_C0, operators, context, encoders[0], relin_key, scale, slot_count);
                     
                        DEBUG_LOG(DEBUG_DEPTH, " > Depth L_2 after sigmoid: " << L_2.depth());
             
                        // ****************** Backpropagation ****************** //

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> E_2(context);
                        operators.negate(L_2, E_2); 
                        heongpu::Plaintext<heongpu::Scheme::CKKS>  packed_y_curr(context);
                        encoders[0].encode(packed_y_curr, packed_y[client_id][t*b+item], scale); 

                        while (packed_y_curr.depth() < E_2.depth()) {
                            operators.mod_drop_inplace(packed_y_curr);
                        }

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth packed_y_curr after mod drop: " << packed_y_curr.depth());
                        
                        operators.add_plain_inplace(E_2, packed_y_curr);

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 after add: " << E_2.depth());

                        heongpu::Ciphertext<heongpu::Scheme::CKKS> d(context);

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth d after derivative sigmoid: " << d.depth());

                        if (item == b-1 && t == m-1) {
                            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2_hlf;
                            encoders[0].encode(P_m2_hlf, m2_hlf, scale);
                            while (P_m2_hlf.depth() < E_2.depth()) {
                                operators.mod_drop_inplace(P_m2_hlf); 
                            }
                            
                            operators.multiply_plain_inplace(E_2, P_m2_hlf); 
                            operators.rescale_inplace(E_2);
                        }
                        else
                        {
                            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
                            encoders[0].encode(P_m2, m2, scale);
                            while (P_m2.depth() < E_2.depth()) {
                                    operators.mod_drop_inplace(P_m2); 
                            }
                            operators.multiply_plain_inplace(E_2, P_m2); 
                            operators.rescale_inplace(E_2);
                        }

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 after multiply: " << E_2.depth());

                      
                        // while (E_2.depth() < d.depth()) { // This doesn't run anyway
                        //     operators.mod_drop_inplace(E_2);
                        // }

                        // operators.multiply_inplace(E_2, d); // E_2 Depth 11
                        // operators.relinearize_inplace(E_2, relin_key);
                        // operators.rescale_inplace(E_2);
                        // DEBUG_LOG(DEBUG_DEPTH, " > Depth E_2 after multiply: " << E_2.depth());

                        RR(E_2, h[0], h[1], galois_key, operators, context);

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
                        DEBUG_LOG(DEBUG_DEPTH, " > Depth of interm after multiply: " << d_W_2[client_id].depth());

        
                        while (d_W_2[client_id].depth() < interm.depth()) {
                            operators.mod_drop_inplace(d_W_2[client_id]); // d_W_2[client_id] Depth 12
                        }
                        
                        operators.add_inplace(d_W_2[client_id], interm); // d_W_2[client_id] Depth 12
                        heongpu::Ciphertext<heongpu::Scheme::CKKS> E_1(context);
                            
                        operators.multiply(E_2, C_W[1], E_1); // E_1 Depth 12
                        operators.relinearize_inplace(E_1, relin_key);
                        operators.rescale_inplace(E_1);

                        RIS(E_1, 1, h[2], galois_key, operators, context);

                        Sigmoid(d, U_1, SG_C3, SG_C2, SG_C1, SG_C0, operators, context, encoders[0], relin_key, scale, slot_count);
                        // d Depth 5, U_1 Depth was 2
                       
                        while (d.depth() < E_1.depth()) {
                            operators.mod_drop_inplace(d); // d Depth 12
                        }
                   
                        DEBUG_LOG(DEBUG_DEPTH, " > Depth d after sigmoid: " << d.depth());
                        operators.multiply_inplace(E_1, d); // E_1 Depth 13
                        operators.relinearize_inplace(E_1, relin_key);
                        operators.rescale_inplace(E_1);

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 after multiply: " << E_1.depth());

                        collectiveBootstrapping(E_1, mpc_managers, mpc_manager_server, secret_keys, context, common_seed, N);
                        
                        if (item == b-1 && t == m-1) {
                            heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1_hlf(context);
                            encoders[0].encode(P_m1_hlf, m1_hlf, scale);
                            while(P_m1_hlf.depth() < E_1.depth()) {
                                    operators.mod_drop_inplace(P_m1_hlf);
                            }
                            operators.multiply_plain_inplace(E_1, P_m1_hlf); 
                            operators.rescale_inplace(E_1);
                        }
                        else {
                            encoders[0].encode(P_m1, m1, scale);
                            while(P_m1.depth() < E_1.depth()) {
                                operators.mod_drop_inplace(P_m1); 
                            }
                            operators.multiply_plain_inplace(E_1, P_m1); 
                            operators.rescale_inplace(E_1);
                        }

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 after multiply: " << E_1.depth());

                        RR(E_1, 1, h[0], galois_key, operators, context);

                        DEBUG_LOG(DEBUG_DEPTH, " > Depth E_1 after RR: " << E_1.depth());
                        
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

                    }
                }
            }

            const auto finish_clients{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_per_client{(finish_clients - start_clients)/ N};
            std::cout << "Elapsed time for each client: " << elapsed_seconds_per_client.count() << " seconds." << std::endl;
            const auto start_agg{std::chrono::steady_clock::now()};

            // Aggregate the results
            double ctext_num_curr = number_of_items_in_ctext;
            if (t == m-1) {
                ctext_num_curr = int(items_per_client) % int(number_of_items_in_ctext);
                if (ctext_num_curr == 0)
                    ctext_num_curr = number_of_items_in_ctext;
            }
            std::vector<double> eta(slot_count, 0.1/(b*ctext_num_curr*N));
            heongpu::Plaintext<heongpu::Scheme::CKKS>  eta_P_1(context); heongpu::Plaintext<heongpu::Scheme::CKKS>  eta_P_2(context);
            encoders[0].encode(eta_P_1, eta, scale); encoders[0].encode(eta_P_2, eta, scale);

            // Generate scaled vectors
            heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_1_scaled;        
            encryptor_alice.encrypt(d_W_1_scaled, all_zeros_plain);
            
            heongpu::Ciphertext<heongpu::Scheme::CKKS> d_W_2_scaled;
            encryptor_alice.encrypt(d_W_2_scaled, all_zeros_plain);

            const auto end_agg_singular{std::chrono::steady_clock::now()};

            // Each client holds the dW value in pieces because of batching, sum them up and scale
            for (int j = 0; j < N; j++) {

                while (eta_P_1.depth() < d_W_1[j].depth()) {
                    operators.mod_drop_inplace(eta_P_1);
                }

                operators.multiply_plain_inplace(d_W_1[j], eta_P_1); // d_W_1[j] Depth 16
                operators.rescale_inplace(d_W_1[j]);

                while(d_W_1_scaled.depth() < d_W_1[j].depth()) {
                    operators.mod_drop_inplace(d_W_1_scaled);
                }

                RIS(d_W_1[j], h[0]*h[1], model_repeat, galois_key, operators, context);

                while (eta_P_2.depth() < d_W_2[j].depth()) {
                    operators.mod_drop_inplace(eta_P_2);
                } 
                operators.multiply_plain_inplace(d_W_2[j], eta_P_2); // d_W_2[j] Depth 16
                operators.rescale_inplace(d_W_2[j]);
                
                while(d_W_2_scaled.depth() < d_W_2[j].depth()) {
                    operators.mod_drop_inplace(d_W_2_scaled);
                }
                RIS(d_W_2[j], h[0]*h[1], model_repeat, galois_key, operators, context);
            }

            const auto end_agg_multi{std::chrono::steady_clock::now()};

            for (int j = 0; j < N; j++) {
                operators.add_inplace(d_W_1_scaled, d_W_1[j]); // d_W_1_scaled[j] Depth 16
                operators.add_inplace(d_W_2_scaled, d_W_2[j]); // d_W_2_scaled[j] Depth 16
            }

            while (C_W[0].depth() < d_W_1_scaled.depth()) {
                operators.mod_drop_inplace(C_W[0]);
            }
            while (C_W[1].depth() < d_W_2_scaled.depth()) {
                operators.mod_drop_inplace(C_W[1]);
            }

            operators.add_inplace(C_W[0], d_W_1_scaled);
            operators.add_inplace(C_W[1], d_W_2_scaled);

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_0 before mod drop: " << C_W[0].depth());
            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_1 before mod drop: " << C_W[1].depth());

            // Gotta do what we goota do so that we can bootstrap
            while (C_W[0].depth() < depth_val) {
                operators.mod_drop_inplace(C_W[0]);
            }

            while (C_W[1].depth() < depth_val) {
                operators.mod_drop_inplace(C_W[1]);
            }

            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_0 before bootstrapping: " << C_W[0].depth());
            DEBUG_LOG(DEBUG_DEPTH, " > Depth C_W_1 before bootstrapping: " << C_W[1].depth());

            collectiveBootstrapping(C_W[0], mpc_managers, mpc_manager_server, secret_keys, context, common_seed, N);
            collectiveBootstrapping(C_W[1], mpc_managers, mpc_manager_server, secret_keys, context, common_seed, N);

            printCiphertext(C_W[0], context, encoders[0], mpc_managers, secret_keys, N);

            const auto finish_agg{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds_for_agg{((end_agg_singular - start_agg) + (end_agg_multi - end_agg_singular))/N + (finish_agg - end_agg_multi)};
            total_training_time += elapsed_seconds_for_prep.count() + elapsed_seconds_per_client.count() + elapsed_seconds_for_agg.count();
            std::cout << "Elapsed time for preparation: " << elapsed_seconds_for_prep.count() << " seconds" << std::endl;
            std::cout << "Elapsed time for clients: " << elapsed_seconds_per_client.count() << " seconds" << std::endl;
            std::cout << "Elapsed time for aggregation: " << elapsed_seconds_for_agg.count() << " seconds" << std::endl;
            std::cout << "Total elapsed time for this round: " 
                 << (elapsed_seconds_for_prep.count() + elapsed_seconds_per_client.count() + elapsed_seconds_for_agg.count()) 
                 << " seconds" << std::endl;    
        }
    }

    std::cout << "Total training time: " << total_training_time << " seconds" << std::endl;

    // **********   TRAINING ENDS HERE   ********** //

    // **********   TESTING STARTS HERE   ********** //

    std::cout << "Testing starts!" << std::endl;
    
    int corr = 0;
    for (int t = 0; t < packed_X_t.size(); t++) {
        heongpu::Plaintext<heongpu::Scheme::CKKS>  L_0(context); // *
        encoders[0].encode(L_0, packed_X_t[t], scale); // * L_0 Depth 1
        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_1(context);

        while (L_0.depth() < C_W[0].depth()) {
            operators.mod_drop_inplace(L_0); 
        }
        
        operators.multiply_plain(C_W[0], L_0, U_1); // U_1 Depth 1
        operators.rescale_inplace(U_1);

        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_1 after multiply: " << U_1.depth());

        RIS(U_1, 1, h[0], galois_key, operators, context);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m1;
        encoders[0].encode(P_m1, m1, scale);
        
        while (P_m1.depth() < U_1.depth()) {
            operators.mod_drop_inplace(P_m1); 
        }
       
        operators.multiply_plain_inplace(U_1, P_m1); // U_1 Depth 2
        operators.rescale_inplace(U_1);
        DEBUG_LOG(DEBUG_DEPTH, " > Depth after RIS: " << U_1.depth());
        RR(U_1, 1, h[2], galois_key, operators, context);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_1(context);
        Softplus(L_1, U_1, SP_C3, SP_C2, SP_C1, SP_C0, operators, context, encoders[0], relin_key, scale, slot_count);
        // L_1 Depth 5
        DEBUG_LOG(DEBUG_DEPTH, " > Depth L_1 after softplus: " << L_1.depth());

        heongpu::Ciphertext<heongpu::Scheme::CKKS> U_2(context);

        if (L_1.depth() != C_W[1].depth()) {
            if (L_1.depth() > C_W[1].depth()) {
                while (L_1.depth() > C_W[1].depth()) {
                    operators.mod_drop_inplace(C_W[1]); 
                }
            }
            else if (L_1.depth() < C_W[1].depth()) {
                while (C_W[1].depth() > L_1.depth()) {
                    operators.mod_drop_inplace(L_1); 
                }
            }
        }
       
        operators.multiply(L_1, C_W[1], U_2); 
        operators.relinearize_inplace(U_2, relin_key);
        operators.rescale_inplace(U_2);
        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after multiply: " << U_2.depth());

        RIS(U_2, h[0], h[1], galois_key, operators, context);

        heongpu::Plaintext<heongpu::Scheme::CKKS> P_m2;
        encoders[0].encode(P_m2, m2, scale);
        
        if (U_2.depth() != P_m2.depth()) {
            if (U_2.depth() > P_m2.depth()) {
                while (U_2.depth() > P_m2.depth()) {
                    operators.mod_drop_inplace(P_m2); 
                }
            }
            else if (U_2.depth() < P_m2.depth()) {
                while (P_m2.depth() > U_2.depth()) {
                    operators.mod_drop_inplace(U_2); 
                }
            }
        }
       
        operators.multiply_plain_inplace(U_2, P_m2); 
        operators.rescale_inplace(U_2);
        DEBUG_LOG(DEBUG_DEPTH, " > Depth U_2 after RIS: " << U_2.depth());
        
        collectiveBootstrapping(U_2, mpc_managers, mpc_manager_server, secret_keys, context, common_seed, N);

        heongpu::Ciphertext<heongpu::Scheme::CKKS> L_2(context);
        Sigmoid(L_2, U_2, SG_C3, SG_C2, SG_C1, SG_C0, operators, context, encoders[0], relin_key, scale, slot_count);
        // L_2 Depth 10
        DEBUG_LOG(DEBUG_DEPTH, " > Depth L_2 after sigmoid: " << L_2.depth());

        std::vector<double> L_2_decrypted_vec;
        L_2_decrypted_vec = collectiveDecryption(L_2, mpc_managers, encoders, secret_keys, context, N);
    
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

    return EXIT_SUCCESS;
}
