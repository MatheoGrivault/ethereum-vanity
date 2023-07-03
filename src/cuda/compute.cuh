#pragma once

#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <iomanip>

#ifndef COMPUTE_CUH
#define COMPUTE_CUH

__global__ void keccak256Hash(const uint8_t* input, uint8_t* output);
__global__ void generatePrivateKey(uint8_t* dev_privateKey);
__global__ void checkAddresses(const uint8_t* privateKeys, int numKeys, const uint8_t* prefix, int prefixSize, bool* results);

#endif











#endif