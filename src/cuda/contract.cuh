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

__global__ void generatesalt(curandState* states,uint64_t* salt); 
__device__ void keccak_hash_compute(BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch);
__global__ void computeContractAdresse(uint64_t* salts, uint8_t* deploymentAddress, size_t deploymentAddressLen, uint8_t* bytecode, size_t bytecodeLen, uint8_t* contractAddresses, size_t contractAddressLen);