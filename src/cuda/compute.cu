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

__global__ void keccak256Hash(const uint8_t* input, uint8_t* output){
    // Tableau de coefficients de rotation pour l'algorithme Keccak
    const int R[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
    };

    // Tailles des états de l'algorithme Keccak
    const int stateSize = 1600;
    const int laneSize = 8;

    // Calcul des indices de thread et de bloc
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Calcul de l'index global dans le tableau de sortie
    int idx = bid * blockDim.x + tid;

    // Variables de l'algorithme Keccak
    uint8_t state[stateSize] = {0};
    uint8_t temp[laneSize] = {0};

    // Copie des données d'entrée dans l'état
    if (idx < laneSize) {
        state[idx] = input[idx];
    }

    // Boucle principale de l'algorithme Keccak
    for (int r = 0; r < 24; r++) {
        // XOR de l'état actuel avec les données d'entrée
        state[tid] ^= input[tid];

        // Permutation des voies de l'état
        for (int i = 0; i < stateSize; i++) {
            int j = (i + R[r]) % stateSize;
            temp[tid] = state[i];
            state[i] = state[j];
            state[j] = temp[tid];
        }
    }

    // Copie du résultat dans le tableau de sortie
    if (idx < laneSize) {
        output[idx] = state[idx];
    }
}

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
        
        // Générer l'adresse Ethereum
        unsigned char addressBytes[65];
        size_t addressSize = sizeof(addressBytes);
        if (secp256k1_ec_pubkey_serialize(ctx, addressBytes, &addressSize, &publicKey, SECP256K1_EC_COMPRESSED) != 1) {
            results[tid] = false;
            secp256k1_context_destroy(ctx);
            continue;
        }
        
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
}
