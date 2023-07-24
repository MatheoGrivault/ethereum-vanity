mod args;

use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering, AtomicU64, AtomicUsize};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use args::{Args, Modes};
use clap::Parser;
use ethers::prelude::k256::ecdsa::SigningKey;
use ethers::types::H160;
use ethers::utils::{secret_key_to_address, get_create2_address_from_hash, to_checksum};
use rand::Rng;

// Data shared between threads
struct ThreadData {
    abort: AtomicBool,
    n_tests: AtomicU64,

    best_n_zero_bytes: AtomicUsize,
}

fn main() {
    let mut args = Args::parse();

    if args.prefix.is_none() && args.suffix.is_none() && !args.zero_bytes {
        println!("No prefix, suffix or zero bytes specified. Use --help for more information.");
        return;
    }

    if args.ignore_case {
        if let Some(prefix) = &args.prefix {
            args.prefix = Some(prefix.to_lowercase());
        }

        if let Some(suffix) = &args.suffix {
            args.suffix = Some(suffix.to_lowercase());
        }
    }

    bruteforce(&args);
}

// Initialize thread pool and start threads
fn bruteforce(args: &Args) {
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().unwrap();
    println!("Started {} threads", rayon::current_num_threads());

    let start_time = Instant::now();
    let thread_data = Arc::new(ThreadData {
        abort: AtomicBool::new(false),
        n_tests: AtomicU64::new(0),

        best_n_zero_bytes: AtomicUsize::new(0),
    });

    // Print tests per second
    {
        let thread_data = Arc::clone(&thread_data);
        
        thread::spawn(move || {
            loop {
                print_status(&thread_data, &start_time);
                
                thread::sleep(Duration::from_secs(1));
            }
        });
    }
    
    // Start threads and wait for them to finish
    rayon::scope(|s| {
        for _ in 0..rayon::current_num_threads() {
            let thread_data = Arc::clone(&thread_data);

            s.spawn(move |_| thread(args, &thread_data, &start_time));
        }
    });
}

// Main bruteforce loop thread
fn thread(args: &Args, thread_data: &ThreadData, start_time: &Instant) {
    let mut rng = rand::thread_rng();

    loop {
        // Abort if another thread is done
        if thread_data.abort.load(Ordering::Relaxed) {
            return;
        }
        
        // Compute input and address
        let mut input = [0; 32];
        rng.fill(&mut input);

        let address = match &args.mode {
            Modes::Account => secret_key_to_address(&SigningKey::from_slice(&input).unwrap()),
            Modes::Contract { deployer_address, bytecode } => get_create2_address_from_hash(*deployer_address, &input, bytecode)
        };

        // Increment number of tests
        thread_data.n_tests.fetch_add(1, Ordering::Relaxed);
        
        // Check prefix and suffix
        if args.prefix.is_some() || args.suffix.is_some() {
            let hex_address = match args.ignore_case {
                true => hex::encode(&address),
                false => to_checksum(&address, None)[2..].to_string()
            };

            if let Some(prefix) = &args.prefix {
                if !hex_address.starts_with(prefix) {
                    continue;
                }
            }
            
            if let Some(suffix) = &args.suffix {
                if !hex_address.ends_with(suffix) {
                    continue;
                }
            }
        }
        
        // Calculate number of zero bytes
        let n_zero_bytes = address.as_bytes().iter().filter(|byte| **byte == 0).count();
        
        if !args.zero_bytes {
            print_address(args, &input, &address, n_zero_bytes);
            
            thread_data.abort.store(true, Ordering::Relaxed);
            return;
        }
        
        // New best address
        if n_zero_bytes > thread_data.best_n_zero_bytes.load(Ordering::Relaxed) {
            thread_data.best_n_zero_bytes.store(n_zero_bytes, Ordering::Relaxed);
            
            print_address(args, &input, &address, n_zero_bytes);
            print_status(thread_data, &start_time);
        }
    }
}

// Print address and corresponding input
fn print_address(args: &Args, input: &[u8; 32], address: &H160, n_zero_bytes: usize) {
    match &args.mode {
        Modes::Account => {
            println!("\x1B[2K\rAddress: {}, private key: 0x{}, zero bytes: {}", to_checksum(address, None), hex::encode(input), n_zero_bytes);
        },
        Modes::Contract { deployer_address: _, bytecode: _ } => {
            println!("\x1B[2K\rAddress: {}, salt: 0x{}, zero bytes: {}", to_checksum(address, None), hex::encode(input), n_zero_bytes);
        }
    }
}

// Print tests per second and elapsed time
fn print_status(thread_data: &ThreadData, start_time: &Instant) {
    print!("\x1B[2K\r{} tests/s, {:?}", thread_data.n_tests.load(Ordering::Relaxed), start_time.elapsed());
    stdout().flush().unwrap();
    thread_data.n_tests.store(0, Ordering::Relaxed);
}