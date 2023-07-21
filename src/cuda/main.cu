#include <iostream>
#include <string>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "CLI11/include/CLI/CLI.hpp"
#include "compute.cuh"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

enum class Command {
    Account,
    Contract,
    Help
};

struct Options {
    std::string prefix;
    std::string suffix;
    bool zeroBytes;
    bool ignoreCase;
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
    
    CLI::App app{"Bruteforce Ethereum addresses"};

    CLI::App *account_subcommand = app.add_subcommand("account", "Bruteforce a private key");
    CLI::App *contract_subcommand = app.add_subcommand("contract", "Bruteforce a CREATE2 salt");

    account_subcommand->add_option("-p,--prefix", options.prefix, "Address prefix");
    account_subcommand->add_option("-s,--suffix", options.suffix, "Address suffix");
    account_subcommand->add_flag("-z,--zero-bytes", options.zeroBytes, "Bruteforce forever until stopped by the user, keeping the address with the most zero bytes");
    account_subcommand->add_flag("-i,--ignore-case", options.ignoreCase, "Ignore case for prefix and suffix");

    account_subcommand->callback([&]() { options.command = Command::Account; });
    contract_subcommand->callback([&]() { options.command = Command::Contract; });

    app.add_flag_function("-v,--version", [](int) { std::cout << "Version 1.0.0" << std::endl; exit(0); }, "Print version and exit");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::cerr << "Error parsing command line options: " << e.what() << std::endl;
        exit(app.exit(e));
    }

    return options;
}

int main(int argc, char* argv[]) {
    Options options = parseOptions(argc, argv);

    switch (options.command) {
        case Command::Account: {
            std::cout<< "Bruteforce a private key" << std::endl;

            const int numKeys = THREADS_PER_BLOCK;
            uint8_t* privateKeys = new uint8_t[numKeys * 32];
            uint8_t* dev_privateKeys;
            cudaMalloc((void**)&dev_privateKeys, numKeys * 32 * sizeof(uint8_t));
            dim3 blockDim(THREADS_PER_BLOCK);
            dim3 gridDim((numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            generatePrivateKey<<<gridDim, blockDim>>>(dev_privateKeys, numKeys);
            cudaMemcpy(privateKeys, dev_privateKeys, numKeys * 32 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaFree(dev_privateKeys);

            bool* results = new bool[numKeys];
            std::string* result_addresses = new std::string[numKeys];

            checkAddresses(privateKeys, numKeys, options.prefix.c_str(), options.suffix.c_str(), results, result_addresses);

            for (int i = 0; i < numKeys; ++i) {
                if (results[i]) {
                    std::cout << "Address with prefix " << options.prefix << " and suffix " << options.suffix << " found for private key " << i << "!" << std::endl;
                    std::cout << "Address: " << result_addresses[i] << std::endl;
                }
            }

            delete[] privateKeys;
            delete[] results;
            delete[] result_addresses;

            break;
        }

        case Command::Contract: {
            // Place your existing contract command logic here
            break;
        }
        default: {
            std::cout << "Invalid command. Valid commands are: account, contract" << std::endl;
            return 1;
        }       
    }

    return 0;
}
