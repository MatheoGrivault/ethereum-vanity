#include "include/config.hpp"
#include "include/keccak.cuh"

__global__ void generatePrivateKey(unsigned char* dev_privateKeys, int numKeys){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int idx = bid * blockDim.x + tid;

    curandState state;
    curand_init(0, idx, 0, &state);

    if (idx < numKeys) {
        for (int i = 0; i < 32; i++) {
            dev_privateKeys[idx * 32 + i] = static_cast<unsigned char>(curand(&state) % 256);
        }
    }
}


std::string createAddressString(const uint8_t* addressBytes, int size){
    std::stringstream ss;
    for(int i = 0; i < size; ++i){
        ss << std::setfill('0') << std::setw(2) << std::hex << static_cast<int>(addressBytes[i]);
    }
    return ss.str();
}

void checkAddresses(const unsigned char* privateKeys, int numKeys, const char* prefix_cstr, const char* suffix_cstr, bool* results, std::string* result_addresses) {
    std::string prefix = std::string(prefix_cstr);
    std::string suffix = std::string(suffix_cstr);

    unsigned char* allPublicKeyBytes = new unsigned char[65 * numKeys];
    unsigned char* hashOutputs = new unsigned char[32 * numKeys];

    // Allocation de la mémoire GPU pour toutes les clés publiques et les sorties de hash
    unsigned char* dev_publicKeyBytes;
    unsigned char* dev_hashOutputs;
    cudaMalloc(&dev_publicKeyBytes, sizeof(unsigned char) * 65 * numKeys);
    cudaMalloc(&dev_hashOutputs, sizeof(unsigned char) * 32 * numKeys);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    for (int tid = 0; tid < numKeys; ++tid) {
        const unsigned char* privateKey = privateKeys + tid * 32;

        if (secp256k1_ec_seckey_verify(ctx, privateKey) != 1) {
            results[tid] = false;
            continue;
        }

        secp256k1_pubkey publicKey;
        if (secp256k1_ec_pubkey_create(ctx, &publicKey, privateKey) != 1) {
            results[tid] = false;
            continue;
        }

        unsigned char publicKeyBytes[65];
        size_t publicKeySize = sizeof(publicKeyBytes);
        if (secp256k1_ec_pubkey_serialize(ctx, publicKeyBytes, &publicKeySize, &publicKey, SECP256K1_EC_COMPRESSED) != 1) {
            results[tid] = false;
            continue;
        }

        memcpy(allPublicKeyBytes + tid * 65, publicKeyBytes, 65);
    }

    // Copier toutes les clés publiques vers la mémoire GPU
    cudaMemcpy(dev_publicKeyBytes, allPublicKeyBytes, sizeof(unsigned char) * 65 * numKeys, cudaMemcpyHostToDevice);

    // Exécuter le kernel CUDA pour hacher toutes les clés publiques
    int threads_per_block = 1024;
    int blocks = (numKeys + threads_per_block - 1) / threads_per_block;
    kernel_keccak_hash<<<blocks, threads_per_block>>>(dev_publicKeyBytes, 65, dev_hashOutputs, numKeys, 32);
    cudaDeviceSynchronize();

    // Récupérer les clés publiques hachées vers la CPU
    cudaMemcpy(hashOutputs, dev_hashOutputs, sizeof(unsigned char) * 32 * numKeys, cudaMemcpyDeviceToHost);

    for (int tid = 0; tid < numKeys; ++tid) {
        unsigned char* hashedPublicKey = hashOutputs + tid * 32;

        unsigned char addressBytes[20];
        memcpy(addressBytes, hashedPublicKey + 12, 20);

        std::string address = createAddressString(addressBytes, 20);  // Assume this function already exists

        bool match = true;
        if (address.substr(0, prefix.size()) != prefix) {
            match = false;
        }

        if (match && !suffix.empty() && address.substr(address.size() - suffix.size()) != suffix) {
            match = false;
        }

        results[tid] = match;
        if (match) {
            result_addresses[tid] = address;
        }
    }

    secp256k1_context_destroy(ctx);
    delete[] allPublicKeyBytes;
    delete[] hashOutputs;

    // Libérer la mémoire GPU
    cudaFree(dev_publicKeyBytes);
    cudaFree(dev_hashOutputs);
}

