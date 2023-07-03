#include <iostream>
#include <string>
#include <time.h>
#include <chrono>
#include <cxxopts.hpp>

#include "compute.cuh"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#include <iostream>
#include <string>
#include <cxxopts.hpp>

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
    int threads;
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

Options parseOptions(int argc, char* argv[]) {
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
            throw cxxopts::OptionException("Print help");
        }

        if (result.count("version")) {
            throw cxxopts::OptionException("Print version");
        }

        if (result.count("command") == 0) {
            throw cxxopts::OptionException("No command specified");
        }

        std::string commandStr = result["command"].as<std::string>();
        options.command = parseCommand(commandStr);
    } 
    
    catch (const cxxopts::OptionException& e) {
        std::cerr << "Error parsing command line options: " << e.what() << std::endl;
        std::cerr << cmdlineOptions.help() << std::endl;
        exit(1);
    }

    return options;
}

void printHelp(const cxxopts::Options& options) {
    std::cout << options.help() << std::endl;
}

int main(int argc, char* argv[]) {
    Options options = parseOptions(argc, argv);

    switch (options.command) {
        case Command::Account:
            // Code pour la commande 'account'
            std::cout << "Bruteforce a private key" << std::endl;
            // Ajoutez le code pour la commande 'account' ici
            break;
        case Command::Contract:
            // Code pour la commande 'contract'
            std::cout << "Bruteforce a CREATE2 salt" << std::endl;
            // Ajoutez le code pour la commande 'contract' ici
            break;
        case Command::Help:
            printHelp(cmdlineOptions);
            break;
    }

    return 0;
}
