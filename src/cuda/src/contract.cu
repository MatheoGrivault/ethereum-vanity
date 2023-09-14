#include "include/config.hpp"
#include "include/keccak.cuh"

#ifndef Contract_Thread
#define Contract_Thread 512
#endif

__global__ void generatesalt(curandState* states, unsigned char* salts) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init((unsigned long long) clock() + index, 0, 0, &states[index]);

    // Générer 32 octets (256 bits) de sel
    for (int i = 0; i < 32; i++) {
        salts[index * 32 + i] = (unsigned char)(curand(&states[index]) % 256);
    }
}


__device__ void keccak_hash_compute(BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch) {
    // In-memory allocation since we are on device side
    const WORD KECCAK_BLOCK_SIZE = (n_outbit >> 3);

    // Iterate over each batch
    for (int i = 0; i < n_batch; ++i) {
        BYTE* cuda_in = in + i * inlen;
        BYTE* cuda_out = out + i * KECCAK_BLOCK_SIZE;
        
        device_keccak_hash(cuda_in, inlen, cuda_out, KECCAK_BLOCK_SIZE);
    }
}   

__global__ void computeContractAdresse(unsigned char* salts, uint8_t* deploymentAddress, size_t deploymentAddressLen, uint8_t* bytecode, size_t bytecodeLen, uint8_t* contractAddresses){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned char salt[32];
    for (int i = 0; i < 32; i++) {
        salt[i] = salts[index * 32 + i];
    }

    uint8_t* data = new uint8_t[deploymentAddressLen + 32]; // La taille du sel est maintenant de 32 octets
    memcpy(data, deploymentAddress, deploymentAddressLen);
    memcpy(data + deploymentAddressLen, salt, 32);

    uint8_t* hash = new uint8_t[32];
    keccak_hash_compute(data, deploymentAddressLen + 32, hash, 256, 1);

    memcpy(contractAddresses + (index * 20), hash + 12, 20);

    delete[] data;
    delete[] hash;
}


__device__ int cuda_memcmp(const void* s1, const void* s2, size_t n) {
    const unsigned char *p1 = (unsigned char*)s1, *p2 = (unsigned char*)s2;

    while(n--) {
        if( *p1 != *p2 ) {
            return *p1 - *p2;
        } else {
            p1++;
            p2++;
        }
    }

    return 0;
}

__device__ bool verifyPrefixAndSuffix(uint8_t* address, uint8_t* prefix, size_t prefixLen, uint8_t* suffix, size_t suffixLen) {
    if (prefixLen > 0 && cuda_memcmp(address, prefix, prefixLen) != 0) {
        return false;
    }

    if (suffixLen > 0 && cuda_memcmp(address + 20 - suffixLen, suffix, suffixLen) != 0) {
        return false;
    }

    return true;
}

__device__ int calculateNumZeroBytes(uint8_t* address) {
    int zeroCount = 0;

    for (int i = 0; i < 20; i++) {
        if (address[i] == 0) {
            zeroCount++;
        } else {
            break;
        }
    }

    return zeroCount;
}

__device__ void generateContractAddress(const uint8_t* deploymentAddress, size_t deploymentAddressLen,
                                       const uint8_t* bytecode, size_t bytecodeLen,
                                       const unsigned char* salt, uint8_t* contractAddress) {
    uint8_t* data = new uint8_t[deploymentAddressLen + 32 + bytecodeLen]; // La taille du sel est de 32 octets
    memcpy(data, deploymentAddress, deploymentAddressLen);
    memcpy(data + deploymentAddressLen, salt, 32);
    memcpy(data + deploymentAddressLen + 32, bytecode, bytecodeLen);

    uint8_t* hash = new uint8_t[32];
    keccak_hash_compute(data, deploymentAddressLen + 32 + bytecodeLen, hash, 256, 1);

    memcpy(contractAddress, hash + 12, 20);

    delete[] data;
    delete[] hash;
}


__global__ void verifyContractAdresse(const uint8_t* deploymentAddress, size_t deploymentAddressLen,
                                        const uint8_t* bytecode, size_t bytecodeLen,
                                        const unsigned char* salts, size_t numSalts,
                                        uint8_t* validAddresses, int* validAddressesCount,
                                        uint8_t* prefix, size_t prefixLen,
                                        uint8_t* suffix, size_t suffixLen){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < numSalts) {
        uint8_t contractAddress[20];
        generateContractAddress(deploymentAddress, deploymentAddressLen, bytecode, bytecodeLen, salts + index * 32, contractAddress);

        if (verifyPrefixAndSuffix(contractAddress, prefix, prefixLen, suffix, suffixLen)) {
            int nZeroBytes = calculateNumZeroBytes(contractAddress);

            if (nZeroBytes > atomicMax(validAddressesCount, nZeroBytes)) {
                memcpy(validAddresses, contractAddress, 20);
            }
        }
    }
}
