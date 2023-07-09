#include <stdio.h>
#include <stdlib.h>
#include "keccak.cuh"

int main() {
    // Définissez vos variables d'entrée et de sortie
    BYTE input[] = { 'a', 'b', 'c' };
    WORD input_length = 3;
    BYTE output[32]; // Utilisation d'un hachage de 256 bits (32 octets)

    // Effectuez le hachage Keccak CUDA
    mcm_cuda_keccak_hash_batch(input, input_length, output, 256, 1);

    // Affichez le résultat du hachage
    printf("Hash output: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");

    return 0;
}
