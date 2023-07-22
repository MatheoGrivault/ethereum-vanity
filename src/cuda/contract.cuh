#pragma once

#ifndef CONTRACT_CUH
#define CONTRACT_CUH



#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "keccak.cuh"

#ifndef Contract_Thread
#define Contract_Thread 512
#endif

// Generate salt
__global__ void generatesalt(curandState* states, uint64_t* salt);

// Keccak hash computation
__device__ void keccak_hash_compute(BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch);

// Compute contract address
__global__ void computeContractAdresse(uint64_t* salts, uint8_t* deploymentAddress, size_t deploymentAddressLen, uint8_t* bytecode, size_t bytecodeLen, uint8_t* contractAddresses, size_t contractAddressLen);

// Function to compare two memory regions.
__device__ int cuda_memcmp(const void* s1, const void* s2, size_t n)

// Function to verify prefix and suffix of the address.
__device__ bool verifyPrefixAndSuffix(uint8_t* address, size_t addressLen, uint8_t* prefix, size_t prefixLen, uint8_t* suffix, size_t suffixLen);

// Function to calculate number of zero bytes in the address.
__device__ int calculateNumZeroBytes(uint8_t* address);

// Function to generate contract address using deployment address, bytecode and salt.
__device__ void generateContractAddress(const uint8_t* deploymentAddress, size_t deploymentAddressLen,
                                        const uint8_t* bytecode, size_t bytecodeLen,
                                        uint64_t salt, uint8_t* contractAddress);

// Verify contract address
__global__ void verifyContractAdresse(const uint8_t* deploymentAddress, size_t deploymentAddressLen,
                                        const uint8_t* bytecode, size_t bytecodeLen,
                                        uint64_t* salts, size_t numSalts,
                                        uint8_t* validAddresses, size_t validAddressesLen,
                                        int* validAddressesCount);


#endif