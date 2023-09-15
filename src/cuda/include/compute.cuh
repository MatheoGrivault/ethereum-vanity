#pragma once

#include "include/config.hpp"
#ifndef COMPUTE_CUH
#define COMPUTE_CUH

__global__ void generatePrivateKey(unsigned char* dev_privateKeys, int numKeys);
void checkAddresses(const uint8_t* privateKeys, int numKeys, const char* prefix_cstr, const char* suffix_cstr, bool* results, std::string* result_addresses);
std::string createAddressString(const uint8_t* addressBytes, int size);

#endif
 