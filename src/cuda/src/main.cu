#include "include/config.hpp"

int THREADS_PER_BLOCK;
#define TRYTES_PER_TRIT 50000

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

void checkCudaError(cudaError_t error, const char* functionName) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error (" << functionName << "): " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE); // Quitte le programme en cas d'erreur
    }
}


int main(int argc, char* argv[]) {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);   

    int device(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    THREADS_PER_BLOCK = maxThreadsPerBlock;

    Options options = parseOptions(argc, argv);
    std::cout << "Start using " << THREADS_PER_BLOCK << " threads per block\n" << std::endl;
    switch (options.command) {
        case Command::Account: {
            std::cout << "Bruteforce a private key\n" << std::endl;

            const int numKeys = THREADS_PER_BLOCK;
            unsigned char* privateKeys = new unsigned char[numKeys * 32];
            unsigned char* dev_privateKeys;
            cudaMalloc((void**)&dev_privateKeys, numKeys * 32 * sizeof(unsigned char));

            unsigned char* dev_validPrivateKeys;
            int* dev_validCount;
            cudaMalloc((void**)&dev_validPrivateKeys, numKeys * 32 * sizeof(unsigned char));
            cudaMalloc((void**)&dev_validCount, sizeof(int));

            unsigned char* validPrivateKeys = new unsigned char[numKeys * 32];
            int* validCount = new int(0);

            dim3 blockDim(THREADS_PER_BLOCK);
            dim3 gridDim((numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

            while (true) {
                generatePrivateKey<<<gridDim, blockDim>>>(dev_privateKeys, numKeys);
                cudaDeviceSynchronize();
                unsigned char* host_privateKeys = new unsigned char[numKeys * 32];
                cudaMemcpy(host_privateKeys, dev_privateKeys, numKeys * 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                    break;  // Sortir de la boucle en cas d'erreur CUDA
                }

                bool* results = new bool[numKeys];
                std::string* result_addresses = new std::string[numKeys];

                checkAddresses(privateKeys, numKeys, options.prefix.c_str(), options.suffix.c_str(), results, result_addresses);

                for (int i = 0; i < numKeys; ++i) {
                    if (results[i]) {
                        std::cout << "Address with prefix " << options.prefix << " and suffix " << options.suffix << " found for private key " << i << "!" << std::endl;
                        std::cout << "Address: " << result_addresses[i] << std::endl;
                    }
                }

                delete[] results;
                delete[] result_addresses;

                options.loopCount += 1;
                if (options.loop == true) {
                    if (options.loopCount % TRYTES_PER_TRIT == 0) {
                        std::cout << "Loop number: " << options.loopCount << std::endl;
                        bool decision = userprompt();
                        if (decision == true) {
                            continue;
                        } else {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }

            // Libération de la mémoire à la fin
            delete[] privateKeys;
            delete[] validPrivateKeys;
            delete validCount;
            cudaFree(dev_privateKeys);
            cudaFree(dev_validPrivateKeys);
            cudaFree(dev_validCount);

            break;
        }


        case Command::Contract: {
            std::cout << "Bruteforce a CREATE2 salt\n" << std::endl;

            const int numSalts = THREADS_PER_BLOCK;
            const int saltSize = 32;
            unsigned char* h_salts = new unsigned char[numSalts * saltSize];

            while (true) {
                
                unsigned char* dev_salts;
                curandState* devStates;

                cudaMalloc((void**)&devStates, numSalts * sizeof(curandState));
                checkCudaError(cudaGetLastError(), "cudaMalloc for devStates");
                cudaMalloc((void**)&dev_salts, numSalts * saltSize);
                checkCudaError(cudaGetLastError(), "cudaMalloc for dev_salts");

                dim3 blockDim(THREADS_PER_BLOCK);
                dim3 gridDim((numSalts + blockDim.x - 1) / blockDim.x);

                generatesalt<<<gridDim, blockDim>>>(devStates, dev_salts);
                checkCudaError(cudaGetLastError(), "generatesalt kernel launch");

                cudaMemcpy(h_salts, dev_salts, numSalts * saltSize, cudaMemcpyDeviceToHost);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from dev_salts to h_salts");

                std::string deploymentAddressStr = options.deployerAddress;
                std::string bytecodeStr = options.Bytecode;

                size_t deploymentAddressLen = deploymentAddressStr.size() / 2;
                size_t bytecodeLen = bytecodeStr.size() / 2;

                unsigned char* deploymentAddress = new unsigned char[deploymentAddressLen];
                unsigned char* bytecode = new unsigned char[bytecodeLen];

                for (size_t i = 0; i < deploymentAddressLen; i++) {
                    deploymentAddress[i] = std::stoul(deploymentAddressStr.substr(i*2, 2), nullptr, 16);
                }

                for (size_t i = 0; i < bytecodeLen; i++) {
                    bytecode[i] = std::stoul(bytecodeStr.substr(i*2, 2), nullptr, 16);
                }

                size_t prefixLen = options.prefix.size() / 2;
                size_t suffixLen = options.suffix.size() / 2;

                unsigned char* prefix = new unsigned char[prefixLen];
                unsigned char* suffix = new unsigned char[suffixLen];

                for (size_t i = 0; i < prefixLen; i++) {
                    prefix[i] = std::stoul(options.prefix.substr(i*2, 2), nullptr, 16);
                }

                for (size_t i = 0; i < suffixLen; i++) {
                    suffix[i] = std::stoul(options.suffix.substr(i*2, 2), nullptr, 16);
                }

                unsigned char* d_deploymentAddress;
                unsigned char* d_bytecode;
                unsigned char* d_prefix;
                unsigned char* d_suffix;

                cudaMalloc(&d_deploymentAddress, deploymentAddressLen * sizeof(unsigned char));
                checkCudaError(cudaGetLastError(), "cudaMalloc for d_deploymentAddress");
                cudaMalloc(&d_bytecode, bytecodeLen * sizeof(unsigned char));
                checkCudaError(cudaGetLastError(), "cudaMalloc for d_bytecode");
                cudaMalloc(&d_prefix, prefixLen * sizeof(unsigned char));
                checkCudaError(cudaGetLastError(), "cudaMalloc for d_prefix");
                cudaMalloc(&d_suffix, suffixLen * sizeof(unsigned char));
                checkCudaError(cudaGetLastError(), "cudaMalloc for d_suffix");

                cudaMemcpy(d_deploymentAddress, deploymentAddress, deploymentAddressLen * sizeof(unsigned char), cudaMemcpyHostToDevice);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from deploymentAddress to d_deploymentAddress");
                cudaMemcpy(d_bytecode, bytecode, bytecodeLen * sizeof(unsigned char), cudaMemcpyHostToDevice);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from bytecode to d_bytecode");
                cudaMemcpy(d_prefix, prefix, prefixLen * sizeof(unsigned char), cudaMemcpyHostToDevice);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from prefix to d_prefix");
                cudaMemcpy(d_suffix, suffix, suffixLen * sizeof(unsigned char), cudaMemcpyHostToDevice);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from suffix to d_suffix");

                unsigned char* dev_validAddress;
                int* dev_validAddressesCount;
                cudaMalloc((void**)&dev_validAddress, 20 * sizeof(unsigned char));
                checkCudaError(cudaGetLastError(), "cudaMalloc for dev_validAddress");
                cudaMalloc((void**)&dev_validAddressesCount, sizeof(int));
                checkCudaError(cudaGetLastError(), "cudaMalloc for dev_validAddressesCount");

                cudaMemset(dev_validAddressesCount, 0, sizeof(int));
                checkCudaError(cudaGetLastError(), "cudaMemset for dev_validAddressesCount");

                verifyContractAdresse<<<gridDim, blockDim>>>(d_deploymentAddress, deploymentAddressLen, d_bytecode, bytecodeLen,
                                                            dev_salts, numSalts, dev_validAddress, dev_validAddressesCount,
                                                            d_prefix, prefixLen, d_suffix, suffixLen);
                checkCudaError(cudaGetLastError(), "verifyContractAdresse kernel launch");
                cudaDeviceSynchronize();
                unsigned char* validAddress = new unsigned char[20];  
                int* validAddressesCount = new int(0);

                cudaMemcpy(validAddress, dev_validAddress, 20 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from dev_validAddress to validAddress");

                cudaMemcpy(validAddressesCount, dev_validAddressesCount, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError(cudaGetLastError(), "cudaMemcpy from dev_validAddressesCount to validAddressesCount");

                if (*validAddressesCount > 0) {
                    std::cout << "Valid address found:\n";
                    for (int i = 0; i < *validAddressesCount; ++i) {
                        std::cout << "Salt: ";
                        for (int j = 0; j < saltSize; ++j) {
                            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)h_salts[i * saltSize + j];
                        }
                        std::cout << std::dec << " Address: ";
                        for (int j = 0; j < 20; ++j) {
                            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)validAddress[j];
                        }
                        std::cout << std::dec << std::endl;
                    }
                }
                options.loopCount += 1;
                if (options.loop == true) {
                    if (options.loopCount % TRYTES_PER_TRIT == 0) {
                        std::cout << "Loop number: " << options.loopCount << std::endl;
                        bool decision = userprompt();
                        if (decision == true) {
                            continue;
                        } else {
                            break;
                        }
                    }
                } 
                else {
                    break;
                }

                delete[] validAddress;
                delete validAddressesCount;

                cudaFree(dev_salts);
                cudaFree(dev_validAddress);
                cudaFree(dev_validAddressesCount);
                cudaFree(devStates);

                delete[] deploymentAddress;
                delete[] bytecode;
                delete[] prefix;
                delete[] suffix;
            }

            delete[] h_salts;

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
