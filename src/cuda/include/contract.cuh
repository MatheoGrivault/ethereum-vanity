#pragma once

#ifndef CONTRACT_CUH
#define CONTRACT_CUH

#include "include/config.hpp"

#ifndef Contract_Thread
#define Contract_Thread 512
#endif

#ifndef TYPE
#define TYPE 
typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;
#endif


// Generate salt
__global__ void generatesalt(curandState* states, unsigned char* salts);

// Keccak hash computation
__device__ void keccak_hash_compute(BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch);

// Compute contract address
__global__ void computeContractAdresse(unsigned char* salts, unsigned char* deploymentAddress, size_t deploymentAddressLen, unsigned char* bytecode, size_t bytecodeLen, unsigned char* contractAddresses);

// Function to compare two memory regions.
__device__ int cuda_memcmp(const void* s1, const void* s2, size_t n);

// Function to verify prefix and suffix of the address.
__device__ bool verifyPrefixAndSuffix(unsigned char* address, unsigned char* prefix, size_t prefixLen, unsigned char* suffix, size_t suffixLen);

// Function to calculate number of zero bytes in the address.
__device__ int calculateNumZeroBytes(unsigned char* address);

// Function to generate contract address using deployment address, bytecode and salt.
__device__ __device__ void generateContractAddress(const unsigned char* deploymentAddress, size_t deploymentAddressLen,
                                       const unsigned char* bytecode, size_t bytecodeLen,
                                       const unsigned char* salt, unsigned char* contractAddress);

// Verify contract address
__global__ void verifyContractAdresse(const unsigned char* deploymentAddress, size_t deploymentAddressLen,
                                        const unsigned char* bytecode, size_t bytecodeLen,
                                        const unsigned char* salts, size_t numSalts,
                                        unsigned char* validAddresses, int* validAddressesCount,
                                        unsigned char* prefix, size_t prefixLen,
                                        unsigned char* suffix, size_t suffixLen);
__device__ void keccak_hash_compute(BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch);


#endif