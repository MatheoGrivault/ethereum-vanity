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

#include "compute.cuh"
#include "keccak.cuh"

#include <cstdint>


__global__ void generatePrivateKey(uint8_t* dev_privateKeys, int numKeys){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int idx = bid * blockDim.x + tid;

    curandState state;
    curand_init(0, idx, 0, &state);

    // Each thread should generate a full 32-byte private key
    if (idx < numKeys) {
        for (int i = 0; i < 32; i++) {
            dev_privateKeys[idx*32 + i] = curand(&state) % 256;
        }
    }
}

void checkAddresses(const uint8_t* privateKeys, int numKeys, const uint8_t* prefix, int prefixSize, bool* results) {
    uint8_t* dev_input;
    uint8_t* dev_output;
    cudaMalloc((void**)&dev_input, sizeof(uint8_t)*65);
    cudaMalloc((void**)&dev_output, sizeof(uint8_t)*32);

    for (int tid = 0; tid < numKeys; ++tid) {
        secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
        
        // Récupérer la clé privée pour le thread actuel
        const uint8_t* privateKey = privateKeys + tid * 32;
        
        // Vérifier la clé privée
        if (secp256k1_ec_seckey_verify(ctx, privateKey) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }
        
        // Générer la clé publique
        secp256k1_pubkey publicKey;
        if (secp256k1_ec_pubkey_create(ctx, &publicKey, privateKey) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }
        
        // Serialiser la clé publique
        unsigned char publicKeyBytes[65];
        size_t publicKeySize = sizeof(publicKeyBytes);
        if (secp256k1_ec_pubkey_serialize(ctx, publicKeyBytes, &publicKeySize, &publicKey, SECP256K1_EC_COMPRESSED) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }
        
        // Copier les données sur le GPU
        cudaMemcpy(dev_input, publicKeyBytes, sizeof(uint8_t)*65, cudaMemcpyHostToDevice);

        // Utiliser mcm_cuda_keccak_hash_batch pour le calcul de hash
        mcm_cuda_keccak_hash_batch(dev_input, publicKeySize, dev_output, 256, 1);

        // Récupérer le résultat sur le CPU
        uint8_t hashedPublicKey[32];
        cudaMemcpy(hashedPublicKey, dev_output, sizeof(uint8_t)*32, cudaMemcpyDeviceToHost);

        // Prendre les 20 derniers octets du hachage comme adresse Ethereum
        uint8_t addressBytes[20];
        memcpy(addressBytes, hashedPublicKey + 12, 20);

        // Comparer l'adresse avec le préfixe spécifié
        bool match = true;
        for (int i = 0; i < prefixSize; ++i) {
            if (addressBytes[i] != prefix[i]) {
                match = false;
                break;
            }
        }
        
        results[tid] = match;
        
        secp256k1_context_destroy(ctx);
    }

    cudaFree(dev_input);
    cudaFree(dev_output);
}
