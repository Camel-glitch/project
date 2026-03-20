#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include <curand_kernel.h>
#include "gamma.h"


// Monte Carlo simulation kernel
__global__ void MC_euler(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* sum, int n) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Dynamic shared memory for reduction
    extern __shared__ float R1s[]; 
    // Add variance computation for confidence interval if needed (not implemented here)

    float payoff = 0.0f;

    // Only active threads compute the simulation
    if (idx < n) {
        curandState localState = state[idx];
        float2 G; 
        float S = S_0;
        float v = v_0;
        
        // Simulation of price and variance up to time N 
        for (int i = 0; i < N; i++) {
            G = curand_normal2(&localState);
            // Euler discretization
            S = S * (1.0f + r * dt + sqrtf(v * dt) * (rho * G.x + sqrtf(1.0f - rho * rho) * G.y)); 
            // Full truncation / reflective boundary for variance
            v = fabsf(v + k * (theta - v) * dt + sigma * sqrtf(v * dt) * G.x);
        }

        payoff = fmaxf(0.0f, S - K);
        
        // Save state back to global memory for future kernel calls
        state[idx] = localState;
    }

    // All threads write to shared memory (inactive threads write 0.0f)
    R1s[threadIdx.x] = payoff;
    __syncthreads();

    // Parallel reduction in shared memory (Requires blockDim.x to be a power of 2)
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            R1s[threadIdx.x] += R1s[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Thread 0 of each block adds its block's total to the global sum safely
    if (threadIdx.x == 0) {
        atomicAdd(sum, R1s[0]);
    }
}
