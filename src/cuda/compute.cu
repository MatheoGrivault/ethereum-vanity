#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif


#include <string>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <iomanip>
#include <secp256k1.h>
#include <secp256k1_recovery.h>


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

std::string hashCompute (std::string& plaintext, std::string hash){
    const int inputSize = message.size() * sizeof(uint8_t);
    const int outputSize = hash.size() * sizeof(uint8_t);

    uint8_t* dev_input;
    uint8_t* dev_output;

    cudaMalloc((void**)&dev_input, inputSize);
    cudaMalloc((void**)&dev_output, outputSize);

    cudaMemcpy(dev_input, message.data(), inputSize, cudaMemcpyHostToDevice);

    // Définition de la configuration des blocs et des threads
    dim3 blockDim(256); // Nombre de threads par bloc
    dim3 gridDim(1); // Nombre de blocs

    // Exécution du kernel CUDA
    keccak256Hash<<<gridDim, blockDim>>>(dev_input, dev_output);

    // Copie des résultats depuis le GPU vers le CPU
    cudaMemcpy(hash.data(), dev_output, outputSize, cudaMemcpyDeviceToHost);

    // Libération de la mémoire sur le GPU
    cudaFree(dev_input);
    cudaFree(dev_output);

    return hash;
}


bool checkAdress(std::string& adress, std::string& hash){
    
}