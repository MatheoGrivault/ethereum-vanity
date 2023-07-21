#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <secp256k1.h>
#include <secp256k1_recovery.h>

#include "keccak.cuh"

#include <cstdint>

__global__ void generatePrivateKey(uint8_t* dev_privateKeys, int numKeys){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int idx = bid * blockDim.x + tid;

    curandState state;
    curand_init(0, idx, 0, &state);

    if (idx < numKeys) {
        for (int i = 0; i < 32; i++) {
            dev_privateKeys[idx*32 + i] = curand(&state) % 256;
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

void checkAddresses(const uint8_t* privateKeys, int numKeys, const char* prefix_cstr, const char* suffix_cstr, bool* results, std::string* result_addresses) {
    std::string prefix = std::string(prefix_cstr);
    std::string suffix = std::string(suffix_cstr);
    
    uint8_t* dev_input;
    uint8_t* dev_output;
    cudaMalloc((void**)&dev_input, sizeof(uint8_t)*65);
    cudaMalloc((void**)&dev_output, sizeof(uint8_t)*32);

    for (int tid = 0; tid < numKeys; ++tid) {
        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

        const uint8_t* privateKey = privateKeys + tid * 32;

        if (secp256k1_ec_seckey_verify(ctx, privateKey) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }

        secp256k1_pubkey publicKey;
        if (secp256k1_ec_pubkey_create(ctx, &publicKey, privateKey) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }

        unsigned char publicKeyBytes[65];
        size_t publicKeySize = sizeof(publicKeyBytes);
        if (secp256k1_ec_pubkey_serialize(ctx, publicKeyBytes, &publicKeySize, &publicKey, SECP256K1_EC_COMPRESSED) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }

        cudaMemcpy(dev_input, publicKeyBytes, sizeof(uint8_t)*65, cudaMemcpyHostToDevice);

        mcm_cuda_keccak_hash_batch(dev_input, publicKeySize, dev_output, 256, 1);

        uint8_t hashedPublicKey[32];
        cudaMemcpy(hashedPublicKey, dev_output, sizeof(uint8_t)*32, cudaMemcpyDeviceToHost);

        uint8_t addressBytes[20];
        memcpy(addressBytes, hashedPublicKey + 12, 20);

        std::string address = createAddressString(addressBytes, 20);

        bool match = true;
        // Check prefix
        if (address.substr(0, prefix.size()) != prefix) {
            match = false;
        }
        // Check suffix
        if (match && !suffix.empty() && address.substr(address.size() - suffix.size()) != suffix) {
            match = false;
        }

        results[tid] = match;
        if (match) {
            result_addresses[tid] = address;
        }

        secp256k1_context_destroy(ctx);
    }

    cudaFree(dev_input);
    cudaFree(dev_output);
}
