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

    app.add_option("-p,--prefix", options.prefix, "Address prefix");
    app.add_option("-s,--suffix", options.suffix, "Address suffix");
    app.add_flag("-z,--zero-bytes", options.zeroBytes, "Bruteforce forever until stopped by the user, keeping the address with the most zero bytes");
    app.add_flag("-i,--ignore-case", options.ignoreCase, "Ignore case for prefix and suffix");

    CLI::App *account_subcommand = app.add_subcommand("account", "Bruteforce a private key");
    CLI::App *contract_subcommand = app.add_subcommand("contract", "Bruteforce a CREATE2 salt");

    account_subcommand->callback([&]() { options.command = Command::Account; });
    contract_subcommand->callback([&]() { options.command = Command::Contract; });

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

            // Nombre de clés privées à générer et à vérifier
            const int numKeys = THREADS_PER_BLOCK;

            // Allocation de la mémoire sur le CPU pour stocker les clés privées
            uint8_t* privateKeys = new uint8_t[numKeys * 32];

            // Allocation de la mémoire sur le GPU
            uint8_t* dev_privateKeys;
            cudaMalloc((void**)&dev_privateKeys, numKeys * 32 * sizeof(uint8_t));

            // Définition de la configuration des blocs et des threads
            dim3 blockDim(THREADS_PER_BLOCK); // Nombre de threads par bloc
            dim3 gridDim((numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK); // Nombre de blocs

            // Exécution du kernel CUDA pour générer les clés privées
            generatePrivateKey<<<gridDim, blockDim>>>(dev_privateKeys, numKeys);

            // Copie du résultat depuis le GPU vers le CPU
            cudaMemcpy(privateKeys, dev_privateKeys, numKeys * 32 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

            // Libération de la mémoire sur le GPU
            cudaFree(dev_privateKeys);

            // Préfixe de l'adresse Ethereum à rechercher
            std::string prefix = options.prefix.empty() ? "0x" : options.prefix;
            const int prefixSize = prefix.size() / 2; // Taille du préfixe en octets

            // Tableau pour stocker les résultats de vérification
            bool* results = new bool[numKeys];

            // Print private keys
            std::cout << "Generated private keys:" << std::endl;
            for (int i = 0; i < numKeys; ++i) {
                std::stringstream ss;
                for (int j = 0; j < 32; ++j) {
                    ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(privateKeys[i * 32 + j]);
                }
                std::cout << ss.str() << std::endl;
            }

            // Vérification des adresses Ethereum pour toutes les clés privées générées
            checkAddresses(privateKeys, numKeys, reinterpret_cast<const uint8_t*>(prefix.data()), prefixSize, results);

            // Vérification du résultat
            for (int i = 0; i < numKeys; ++i) {
                if (results[i]) {
                    std::cout << "Address with prefix " << prefix << " found for private key " << i << "!" << std::endl;
                }
            }

            // Libération de la mémoire sur le CPU
            delete[] privateKeys;
            delete[] results;

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
