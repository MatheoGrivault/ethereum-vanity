# Ethereum Vanity
Generate vanity account and contract addresses with a specific prefix, suffix or addresses with as many zero bytes as possible by bruteforcing a private key or a CREATE2 salt.

Addresses with zero bytes consume less gas:
- 4 gas for zero bytes
- 68 gas for non-zero bytes (17x more)

These gas savings only apply to calldata (i.e msg.data).
## Usage
```
Generate account and contract addresses by bruteforcing a private key or a CREATE2 salt

Usage: ethereum_vanity [OPTIONS] <COMMAND>

Commands:
  account   Bruteforce a private key
  contract  Bruteforce a CREATE2 salt
  help      Print this message or the help of the given subcommand(s)

Options:
  -p, --prefix <PREFIX>    Address prefix
  -s, --suffix <SUFFIX>    Address suffix
  -z, --zero-bytes         Bruteforce forever until stopped by the user, keeping the address with the most zero bytes
  -i, --ignore-case        Ignore case for prefix and suffix
  -t, --threads <THREADS>  Number of threads to use, or 0 to use all logical cores [default: 0]
  -h, --help               Print help
  -V, --version            Print version
```
## Account mode
```
Bruteforce a private key

Usage:  ethereum_vanity account

Options:
-h, --help Print help
```
### Example
Generate account address with prefix "**0xabcd**" (ignore-case)
```
./ethereum_vanity -p abcd -i account
Started 8 threads
Address: 0xabCD0fc318dE8b3EE2Eb1351150bcd28B13413d0, private key: 0x68cb4694bebf4f42bf54353c5fce6781a50b8ed6cff228033cceb20f3dc6c9de, zero bytes: 0
```
## Contract mode
```
Bruteforce a CREATE2 salt

Usage: ethereum_vanity contract --deployer-address <DEPLOYER_ADDRESS> --bytecode <BYTECODE>

Options:
  -d, --deployer-address <DEPLOYER_ADDRESS>  Address of the deployer
  -b, --bytecode <BYTECODE>                  Bytecode of the contract
  -h, --help                                 Print help
```
### Example
Generate contract address with suffix "**eF**" (case-sensitive) with as many zero bytes as possible
```
./ethereum_vanity -s eF -z contract -d 0x0000111122223333444455556666777788889999 -b 0x1234
Started 8 threads
Address: 0x886D92b7Baf1fbA3EA842e458C891d2d479800eF, salt: 0x0c83f322c3c7c4d05afb7f623b8ed7e34d6bc0275ac00d84f7bd664e1460f144, zero bytes: 1
Address: 0x6Cb2A39231F9c0E21Fa3491600B8e80006d08deF, salt: 0x3b0b1388334bbcaa9b5e88050fe5bb266ac241d887edaf9bf02fb31ff74f4599, zero bytes: 2
Address: 0x6dB42F2AF47FD2005E5669476A000088Dcfe6ceF, salt: 0x92be5221785d5828962fdfb2e5a52757ce1b51c53f29a9027ddc6fb0daf21018, zero bytes: 3
1808849 tests/s, 8.000621067s
```
