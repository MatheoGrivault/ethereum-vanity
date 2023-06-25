use clap::{Parser, Subcommand};
use ethers::{types::Address, utils::keccak256};
use regex::Regex;

/// Generate account and contract addresses by bruteforcing a private key or a CREATE2 salt
#[derive(Parser)]
pub struct Args {
    #[command(subcommand)]
    pub mode: Modes,

    /// Address prefix
    #[clap(short, long, value_parser=is_valid_prefix_suffix)]
    pub prefix: Option<String>,

    /// Address suffix
    #[clap(short, long, value_parser=is_valid_prefix_suffix)]
    pub suffix: Option<String>,

    /// Bruteforce forever until stopped by the user, keeping the address with the most zero bytes
    #[clap(short, long)]
    pub zero_bytes: bool,

    /// Ignore case for prefix and suffix
    #[clap(short, long)]
    pub ignore_case: bool,

    /// Number of threads to use, or 0 to use all logical cores
    #[clap(short, long, default_value="0")]
    pub threads: usize
}

#[derive(Subcommand)]
pub enum Modes {
    /// Bruteforce a private key
    Account,

    /// Bruteforce a CREATE2 salt
    Contract {
        /// Address of the deployer
        #[clap(short, long, value_parser=is_valid_address)]
        deployer_address: Address,

        /// Bytecode of the contract
        #[clap(short, long, value_parser=is_valid_bytecode)]
        bytecode: [u8; 32],
    }
}

// Check if address is valid
fn is_valid_address(address: &str) -> Result<Address, String> {
    match Regex::new(r"^[a-fA-F0-9]{40}$") {
        Ok(regex) => {
            let address = address.replace("0x", "");

            if regex.is_match(&address) {
                Ok(Address::from_slice(&hex::decode(&address).unwrap()))
            } else {
                Err("Invalid address".to_string())
            }
        },
        Err(err) => Err(err.to_string())
    }
}

// Check if bytecode is valid and precompute its keccak256 hash
fn is_valid_bytecode(bytecode: &str) -> Result<[u8; 32], String> {
    match Regex::new(r"^[a-fA-F0-9]*$") {
        Ok(regex) => {
            let bytecode = bytecode.replace("0x", "");

            if regex.is_match(&bytecode) {
                Ok(keccak256(hex::decode(&bytecode).unwrap()))
            } else {
                Err("Invalid bytecode".to_string())
            }
        },
        Err(err) => Err(err.to_string())
    }
}

// Check if hex is valid
fn is_valid_prefix_suffix(hex: &str) -> Result<String, String> {
    match Regex::new(r"^[a-fA-F0-9]*$") {
        Ok(regex) => {
            let hex = hex.replace("0x", "");

            if hex.len() > 40 {
                return Err("Prefix or suffix too long. Maximum length is 40.".to_string());
            }

            if regex.is_match(&hex) {
                Ok(hex)
            } else {
                Err("Invalid hex".to_string())
            }
        },
        Err(err) => Err(err.to_string())
    }
}