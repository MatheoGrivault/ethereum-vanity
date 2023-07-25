#include <iostream>
#include <string>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <sstream>  
#include<curand.h>
#include<curand_kernel.h>

#include "CLI11/include/CLI/CLI.hpp"
#ifndef KECCAK_INCLUDE
#define KECCAK_INCLUDE
#include "keccak.cuh"
#endif
#include "compute.cuh"
#include "contract.cuh"
#include "config.h"
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 512
#endif

enum class Command {    
    Account,
    Contract,
    Help
};

struct Options {
    std::string prefix;
    std::string suffix;
    std::string deployerAddress;
    std::string Bytecode;
    bool zeroBytes;
    bool ignoreCase;
    bool loop;
    int loopCount;
    Command command;
};

Command parseCommand(const std::string& commandStr) {
    if (commandStr == "account") {
        return Command::Account;
    } else if (commandStr == "contract") {
        return Command::Contract;
    } else {
        throw CLI::ValidationError("Invalid command. Valid commands are: account, contract");
    }
}

Options parseOptions(int argc, char* argv[]) {
    Options options;
    options.loopCount=1;

    CLI::App app{"Bruteforce Ethereum addresses"};

    CLI::App *account_subcommand = app.add_subcommand("account", "Bruteforce a private key");
    CLI::App *contract_subcommand = app.add_subcommand("contract", "Bruteforce a CREATE2 salt");

    //account command options
    account_subcommand->callback([&]() { options.command = Command::Account; });
    account_subcommand->add_option("-p,--prefix", options.prefix, "Address prefix");
    account_subcommand->add_option("-s,--suffix", options.suffix, "Address suffix");
    account_subcommand->add_flag("-z,--zero-bytes", options.zeroBytes, "Bruteforce forever until stopped by the user, keeping the address with the most zero bytes");
    account_subcommand->add_flag("-i,--ignore-case", options.ignoreCase, "Ignore case for prefix and suffix");
    account_subcommand->add_flag("-l,--loop", options.loop, "Loop through all private keys");

    //contract command options
    contract_subcommand->callback([&]() { options.command = Command::Contract; });
    contract_subcommand->add_option("-p,--prefix", options.prefix, "Contract address prefix");
    contract_subcommand->add_option("-s,--suffix", options.suffix, "Contract address suffix");
    contract_subcommand->add_option("-d,--deployer", options.deployerAddress, "Deployer's address");
    contract_subcommand->add_option("-b,--bytecode", options.Bytecode, "Contract's bytecode");
    contract_subcommand->add_flag("-z,--zero-bytes", options.zeroBytes, "Bruteforce forever until stopped by the user, keeping the address with the most zero bytes");
    contract_subcommand->add_flag("-i,--ignore-case", options.ignoreCase, "Ignore case for prefix and suffix");
    contract_subcommand->add_flag("-l,--loop", options.loop, "Loop through all salts");

    app.add_flag_function("-v,--version", [](int) { std::cout << "Version 1.0.0" << std::endl; exit(0); }, "Print version and exit");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::cerr << "Error parsing command line options: " << e.what() << std::endl;
        exit(app.exit(e));
    }

    return options;
}

bool userprompt(){
    char decision;
    
    std::cout << "Do you want to continue ? (y/n)" << std::endl;
    std::cin >> decision;
    if (decision == 'y' || decision == 'Y') {
        return true;
    } else {
        return false;
    }
}

int main(int argc, char* argv[]) {
    Options options = parseOptions(argc, argv);
    std::cout << "Start using " << THREADS_PER_BLOCK << " threads per block\n" << std::endl;
    switch (options.command) {

        case Command::Account: {
            std::cout<< "Bruteforce a private key\n" << std::endl;
            
            while (true) {
                std::cout << "Loop number: " << options.loopCount << std::endl;
                const int numKeys = THREADS_PER_BLOCK;
                uint8_t* privateKeys = new uint8_t[numKeys * 32];
                uint8_t* dev_privateKeys;
                cudaMalloc((void**)&dev_privateKeys, numKeys * 32 * sizeof(uint8_t));
                
                uint8_t* dev_validPrivateKeys;
                int* dev_validCount;
                cudaMalloc((void**)&dev_validPrivateKeys, numKeys * 32 * sizeof(uint8_t));
                cudaMalloc((void**)&dev_validCount, sizeof(int));

                dim3 blockDim(THREADS_PER_BLOCK);
                dim3 gridDim((numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
                generatePrivateKey<<<gridDim, blockDim>>>(dev_privateKeys, numKeys);
                // New memory copy operations
                uint8_t* validPrivateKeys = new uint8_t[numKeys * 32];
                int* validCount = new int(0);
                cudaMemcpy(validPrivateKeys, dev_validPrivateKeys, numKeys * 32 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(validCount, dev_validCount, sizeof(int), cudaMemcpyDeviceToHost);


                bool* results = new bool[numKeys];
                std::string* result_addresses = new std::string[numKeys];

                checkAddresses(privateKeys, numKeys, options.prefix.c_str(), options.suffix.c_str(), results, result_addresses);

                for (int i = 0; i < numKeys; ++i) {
                    if (results[i]) {
                        std::cout << "Address with prefix " << options.prefix << " and suffix " << options.suffix << " found for private key " << i << "!" << std::endl;
                        std::cout << "Address: " << result_addresses[i] << std::endl;
                    }
                    else {
                        std::cout << "No address with prefix " << options.prefix << " and suffix " << options.suffix << " found for private key " << i << "." << std::endl;
                    }
                }

                delete[] privateKeys;
                delete[] results;
                delete[] result_addresses;

                if (options.loop == true ){
                    options.loopCount += 1;
                    bool decision = userprompt();
                    if (decision == true){
                        continue;
                    }
                    else{
                        break;
                        return 0;
                    }
                    continue;
                }
                else{
                    break;
                }
                
            }
            break;
        }

        case Command::Contract: {
            std::cout << "Bruteforce a CREATE2 salt\n" << std::endl;

            while (true){
                std::cout << "Loop number: " << options.loopCount << std::endl;
                const int numSalts = THREADS_PER_BLOCK;

                // Allocate memory for salts and random states on the device
                uint64_t* dev_salts;
                curandState* devStates;
                cudaMalloc(&devStates, numSalts * sizeof(curandState));
                cudaMalloc((void**)&dev_salts, numSalts * sizeof(uint64_t));

                // Define grid and block dimensions
                dim3 blockDim(THREADS_PER_BLOCK);
                dim3 gridDim((numSalts + blockDim.x - 1) / blockDim.x);

                // Generate salts using the device function
                generatesalt<<<gridDim, blockDim>>>(devStates, dev_salts);

                // Assuming you have a deployment address and bytecode, replace these with the actual values.
                std::string deploymentAddressStr = options.deployerAddress; // replace with actual deployment address
                std::string bytecodeStr = options.Bytecode; // replace with actual bytecode

                size_t deploymentAddressLen = deploymentAddressStr.size() / 2;
                size_t bytecodeLen = bytecodeStr.size() / 2;

                uint8_t* deploymentAddress = new uint8_t[deploymentAddressLen];
                uint8_t* bytecode = new uint8_t[bytecodeLen];

                for (size_t i = 0; i < deploymentAddressLen; i++) {
                    deploymentAddress[i] = std::stoul(deploymentAddressStr.substr(i*2, 2), nullptr, 16);
                }

                for (size_t i = 0; i < bytecodeLen; i++) {
                    bytecode[i] = std::stoul(bytecodeStr.substr(i*2, 2), nullptr, 16);
                }

                // Convert prefix and suffix from string to uint8_t*
                size_t prefixLen = options.prefix.size() / 2;
                size_t suffixLen = options.suffix.size() / 2;

                uint8_t* prefix = new uint8_t[prefixLen];
                uint8_t* suffix = new uint8_t[suffixLen];

                for (size_t i = 0; i < prefixLen; i++) {
                    prefix[i] = std::stoul(options.prefix.substr(i*2, 2), nullptr, 16);
                }

                for (size_t i = 0; i < suffixLen; i++) {
                    suffix[i] = std::stoul(options.suffix.substr(i*2, 2), nullptr, 16);
                }

                // Copying host data to device
                uint8_t* d_deploymentAddress;
                uint8_t* d_bytecode;
                uint8_t* d_prefix;
                uint8_t* d_suffix;

                cudaMalloc(&d_deploymentAddress, deploymentAddressLen * sizeof(uint8_t));
                cudaMalloc(&d_bytecode, bytecodeLen * sizeof(uint8_t));
                cudaMalloc(&d_prefix, prefixLen * sizeof(uint8_t));
                cudaMalloc(&d_suffix, suffixLen * sizeof(uint8_t));

                cudaMemcpy(d_deploymentAddress, deploymentAddress, deploymentAddressLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_bytecode, bytecode, bytecodeLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_prefix, prefix, prefixLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_suffix, suffix, suffixLen * sizeof(uint8_t), cudaMemcpyHostToDevice);

                // Allocate memory on the device for the address verification results
                uint8_t* dev_validAddress;
                int* dev_validAddressesCount;
                cudaMalloc((void**)&dev_validAddress, 20 * sizeof(uint8_t));
                cudaMalloc((void**)&dev_validAddressesCount, sizeof(int));

                verifyContractAdresse<<<gridDim, blockDim>>>(d_deploymentAddress, deploymentAddressLen, d_bytecode, bytecodeLen,
                                                            dev_salts, numSalts, dev_validAddress, dev_validAddressesCount,
                                                            d_prefix, prefixLen, d_suffix, suffixLen);

                // Allocate memory on the host for the address verification results
                uint8_t* validAddress = new uint8_t[20];
                int* validAddressesCount = new int(0);

                // Copy results from device to host
                cudaMemcpy(validAddress, dev_validAddress, 20 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(validAddressesCount, dev_validAddressesCount, sizeof(int), cudaMemcpyDeviceToHost);

                if (*validAddressesCount > 0) {
                    std::cout << "Valid contract address with " << *validAddressesCount << " leading zero bytes found: ";
                    for (int j = 0; j < 20; ++j) {
                        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)validAddress[j];
                    }
                    
                    std::cout << std::endl;
                } else {
                    std::cout << "No valid contract address found." << std::endl;
                }
                if (options.loop == true ){
                        options.loopCount += 1;
                        bool decision = userprompt();
                        if (decision == true){
                            continue;
                        }
                        else{
                            break;
                            return 0;
                        }
                        continue;
                    
                }
                else {
                    break;
                }

                
                // Cleanup memory
                delete[] deploymentAddress;
                delete[] bytecode;
                delete[] validAddress;
                delete validAddressesCount;
                delete[] prefix;
                delete[] suffix;

                cudaFree(d_deploymentAddress);
                cudaFree(d_bytecode);
                cudaFree(devStates);
                cudaFree(dev_salts);
                cudaFree(dev_validAddress);
                cudaFree(dev_validAddressesCount);
                cudaFree(d_prefix);
                cudaFree(d_suffix);

                
            }
            break;
        }

        default: {
            std::cout << "Invalid command. Valid commands are: account, contract" << std::endl;
            std::cout << "Use --help for more information.\n" << std::endl;
            return 1;
        }       
    }

    return 0;
}
