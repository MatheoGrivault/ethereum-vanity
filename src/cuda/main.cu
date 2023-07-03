#include <iostream>
#include <string>
#include <time.h>
#include <chrono>
#include <cxxopts.hpp>

#include "compute.cuh"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

enum class Command{
    Account
    Contract
    Help
};

struct Options {
    std::string prefix;
    std::string suffix; 
    bool zeroBytes;
    bool ignoreCase;
    int threads
};

Command parseCommand(const std::string& commandstr){
    if commandstr == "account"{
        return Command::Account;
    }

    else if commandstr == "contract"{
        return Command::Contract;
    
    }
    else if commandstr == "help"{
        return Command::Help;
    }

    else{
        std::cout<<"Invalid command"<<std::endl;
        return Command::Help;
    }
}

Options parseOptions(int argc, char* argv[]){
    Options options;
    cxxopts::Options options("ethereum_vanity", "Bruteforce Ethereum addresses");

    cmdlineOptions.add_options()
        ("p,prefix", "Address prefix", cxxopts::value<std::string>(options.prefix))
        ("s,suffix", "Address suffix", cxxopts::value<std::string>(options.suffix))
        ("z,zero-bytes", "Bruteforce forever until stopped by the user, keeping the address with the most zero bytes", cxxopts::value<bool>(options.zeroBytes))
        ("i,ignore-case", "Ignore case for prefix and suffix", cxxopts::value<bool>(options.ignoreCase))
        ("t,threads", "Number of threads to use, or 0 to use all logical cores", cxxopts::value<int>(options.threads))
        ("h,help", "Print help")
        ("V,version", "Print version");
    
    cmdlineOptions.parse_positionnal({"command"});

    try {

        auto result = cmdlineOptions.parse(argc, argv);

        if result.count("help"){
            trow cxxopts::OptionException("Print help");
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

int main(int argc, char ** argv[]){

    



    std::cout<<"Starting brute force attack"<<std::endl;

}