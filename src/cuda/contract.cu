#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "keccak.cuh"
#include "config.h"


#ifndef Contract_Thread
#define Contract_Thread 512
#endif

__global__ void generatesalt (curandState* states,uint64_t* salt){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long seed = index +clock64();
    curand_init(seed, index, 0, &states[index]);

    salt[index] = curand(&states[index]);   
}

__device__ void keccak_hash_compute(BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch){
    mcm_cuda_keccak_hash_batch(in, inlen, out, n_outbit, n_batch);
}

__global__ void computeContractAdresse(uint64_t* salts, uint8_t* deploymentAddress, size_t deploymentAddressLen, uint8_t* bytecode, size_t bytecodeLen, uint8_t* contractAddresses, size_t contractAddressLen){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index <Contract_Thread) {
        // Obtenir le "salt" pour ce thread
        uint64_t salt = salts[index];

        // Concaténer le "salt" avec l'adresse de déploiement
        uint8_t* data = new uint8_t[deploymentAddressLen + sizeof(uint64_t)];
        memcpy(data, deploymentAddress, deploymentAddressLen);
        memcpy(data + deploymentAddressLen, &salt, sizeof(uint64_t));

        // Calculer le hash sha3 du "salt" concaténé avec l'adresse de déploiement
        uint8_t* hash = new uint8_t[32];
        keccak_hash_batch_wrapper(data, deploymentAddressLen + sizeof(uint64_t), hash, 256, 1);

        // Libérer la mémoire allouée pour la concaténation
        delete[] data;

        // Utiliser les 20 derniers octets du hash sha3 comme l'adresse de contrat
        memcpy(contractAddresses + (index * contractAddressLen), hash + 12, contractAddressLen);

        // Libérer la mémoire allouée pour le hash sha3
        delete[] hash;
    }
}

__global__ void verifyContractAdresse(const uint8_t* deploymentAddress, size_t deploymentAddressLen,
                                        const uint8_t* bytecode, size_t bytecodeLen,
                                        uint64_t* salts, size_t numSalts,
                                        uint8_t* validAddresses, size_t validAddressesLen){


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = Contract_Thread;

    for (size_t i = tid; i < numSalts; i += numThreads) {
        // Generate contract address using the deployment address, bytecode, and salt
        uint8_t contractAddress[20];
        generateContractAddress(deploymentAddress, deploymentAddressLen, bytecode, bytecodeLen, salts[i], contractAddress);

        // Verify the prefix and suffix of the generated address
        if (verifyPrefixAndSuffix(contractAddress)) {
            // Calculate the number of zero bytes in the generated address
            int nZeroBytes = calculateNumZeroBytes(contractAddress);

            // Check if the address meets the criteria of having the most zero bytes
            if (nZeroBytes > atomicMax(validAddressesCount, nZeroBytes)) {
                // Store the address if it has the most zero bytes so far
                for (int j = 0; j < 20; j++) {
                    validAddresses[j] = contractAddress[j];
                }
            }
        }
    }
}