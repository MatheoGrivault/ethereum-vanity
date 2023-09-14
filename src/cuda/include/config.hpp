#pragma once
#define USE_MD2 1
#define USE_MD5 1
#define USE_SHA1 1
#define USE_SHA256 1

#define CUDA_HASH 1
#define OCL_HASH 0


#include <stdlib.h>
#include <string.h>


extern int THREADS_PER_BLOCK;

#ifndef DEPENDENCIES
#define DEPENDENCIES

#include <random>
#include <fstream>
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
#include <time.h>
#include <chrono>
#include <iomanip>
#include <CLI/CLI.hpp>

#include "compute.cuh"
#include "contract.cuh"
#include "keccak.cuh"
#include "include/keccak.cuh"

#ifndef TYPE
#define TYPE 
typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef unsigned long long LONG;
#endif

#endif

