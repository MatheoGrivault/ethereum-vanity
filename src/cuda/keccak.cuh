#ifndef _KECCAK_CUH_
#define _KECCAK_CUH_

#include <cuda_runtime.h>
#include "config.h"

#define KECCAK_ROUND 24
#define KECCAK_STATE_SIZE 25
#define KECCAK_Q_SIZE 192

typedef struct {
    BYTE sha3_flag;
    WORD digestbitlen;
    LONG rate_bits;
    LONG rate_BYTEs;
    LONG absorb_round;
    int64_t state[KECCAK_STATE_SIZE];
    BYTE q[KECCAK_Q_SIZE];
    LONG bits_in_queue;
} cuda_keccak_ctx_t;

typedef cuda_keccak_ctx_t CUDA_KECCAK_CTX;

__device__ LONG cuda_keccak_leuint64(void *in);
__device__ int64_t cuda_keccak_MIN(int64_t a, int64_t b);
__device__ LONG cuda_keccak_UMIN(LONG a, LONG b);
__device__ void cuda_keccak_extract(cuda_keccak_ctx_t *ctx);
__device__ __forceinline__ LONG cuda_keccak_ROTL64(LONG a, LONG  b);
__device__ void cuda_keccak_permutations(cuda_keccak_ctx_t * ctx);
__device__ void cuda_keccak_absorb(cuda_keccak_ctx_t *ctx, BYTE* in);
__device__ void cuda_keccak_pad(cuda_keccak_ctx_t *ctx);
__device__ void cuda_keccak_init(cuda_keccak_ctx_t *ctx, WORD digestbitlen);
__device__ void cuda_keccak_sha3_init(cuda_keccak_ctx_t *ctx, WORD digestbitlen);
__device__ void cuda_keccak_update(cuda_keccak_ctx_t *ctx, BYTE *in, LONG inlen);
__global__ void kernel_keccak_hash(BYTE* indata, WORD inlen, BYTE* outdata, WORD n_batch, WORD KECCAK_BLOCK_SIZE);
__device__ void device_keccak_hash(BYTE* indata, WORD inlen, BYTE* outdata, WORD KECCAK_BLOCK_SIZE);
__device__ void cuda_keccak_final(cuda_keccak_ctx_t *ctx, BYTE *out);
__device__ void cuda_keccak_sha3_final(cuda_keccak_ctx_t *ctx, BYTE *out);

extern "C" {
    void mcm_cuda_keccak_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_outbit, WORD n_batch);
}


#endif // _KECCAK_CUH_
