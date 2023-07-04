#include <iostream>
#include <string>
#include <time.h>
#include <chrono>
#include <cxxopts.hpp>
#include <iomanip>
#include <sstream>

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
    } else if (commandStr == "help") {
        return Command::Help;
    } else {
        return Command::Help;
    }
}

std::pair<Options, cxxopts::Options> parseOptions(int argc, char* argv[]) {
    Options options;

    cxxopts::Options cmdlineOptions("ethereum_vanity", "Bruteforce Ethereum addresses");

    cmdlineOptions.add_options()
        ("p,prefix", "Address prefix", cxxopts::value<std::string>(options.prefix))
        ("s,suffix", "Address suffix", cxxopts::value<std::string>(options.suffix))
        ("z,zero-bytes", "Bruteforce forever until stopped by the user, keeping the address with the most zero bytes", cxxopts::value<bool>(options.zeroBytes))
        ("i,ignore-case", "Ignore case for prefix and suffix", cxxopts::value<bool>(options.ignoreCase))
        ("h,help", "Print help")
        ("V,version", "Print version");

    cmdlineOptions.parse_positional({"command"});

    try {
        auto result = cmdlineOptions.parse(argc, argv);

        if (result.count("help")) {
            throw std::runtime_error("Print help");
        }

        if (result.count("version")) {
            throw std::runtime_error("Print version");
        }

        if (result.count("command") == 0) {
            throw std::runtime_error("No command specified");
        }

        std::string commandStr = result["command"].as<std::string>();
        options.command = parseCommand(commandStr);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing command line options: " << e.what() << std::endl;
        std::cerr << cmdlineOptions.help() << std::endl;
        exit(1);
    }

    return {options, cmdlineOptions};
}

void printHelp(const cxxopts::Options& options) {
    std::cout << options.help() << std::endl;
}

int main(int argc, char* argv[]) {
    auto [options, cmdlineOptions] = parseOptions(argc, argv);

    switch (options.command) {
        case Command::Account: {
            std::cout << "Bruteforce a private key" << std::endl;
            const int privateKeySize = 32;
            uint8_t privateKey[privateKeySize];

            // Allocation de la mémoire sur le GPU
            uint8_t* dev_privateKey;
            cudaMalloc((void**)&dev_privateKey, privateKeySize * sizeof(uint8_t));

            // Définition de la configuration des blocs et des threads
            dim3 blockDim(THREADS_PER_BLOCK); // Nombre de threads par bloc
            dim3 gridDim(1); // Nombre de blocs

            // Exécution du kernel CUDA
            generatePrivateKey<<<gridDim, blockDim>>>(dev_privateKey);

            // Copie du résultat depuis le GPU vers le CPU
            cudaMemcpy(privateKey, dev_privateKey, privateKeySize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

            // Libération de la mémoire sur le GPU
            cudaFree(dev_privateKey);

            // Conversion du privateKey en hexadécimal
            std::stringstream ss;
            for (int i = 0; i < privateKeySize; i++) {
                ss << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(privateKey[i]);
            }
            std::string privateKeyHex = ss.str();

            std::cout << "Private Key: " << privateKeyHex << std::endl;

            // Vérification de l'adresse Ethereum
            const std::string prefix = "0x"; // Préfixe de l'adresse Ethereum à rechercher
            const int prefixSize = prefix.size() / 2; // Taille du préfixe en octets

            const int numKeys = 1; // Nombre de clés privées à vérifier
            bool results[numKeys]; // Tableau pour stocker les résultats de vérification

            checkAddresses(privateKey, numKeys, reinterpret_cast<const uint8_t*>(prefix.data()), prefixSize, results);

            // Vérification du résultat
            if (results[0]) {
                std::cout << "Address with prefix " << prefix << " found!" << std::endl;
            } else {
                std::cout << "No address with prefix " << prefix << " found." << std::endl;
            }

            break;
        }

        case Command::Contract: {
            // Code pour la commande 'contract'
            std::cout << "Bruteforce a CREATE2 salt" << std::endl;
            // Ajoutez le code pour la commande 'contract' ici
            break;
        }
        case Command::Help: {
            printHelp(cmdlineOptions);
            break;
        }
    }

    return 0;
}
