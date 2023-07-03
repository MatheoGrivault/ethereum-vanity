#pragma once 

#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <iomanip>


#ifndef COMPUTE_CUH 
#define COMPUTE_CUH

__global__ void keccak256Hash(const uint8_t* input, uint8_t* output);
std::string hashCompute (std::string& plaintext, std::string hash);












#endif