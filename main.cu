#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include "gamma.h"
#include "utils.h"
#include "MC_euler.h"
#include "MC_exact.h"
#include "MC_almost.h"

int main(void) {
    int NTPB = 512;
    int NB = 512;
    long long n =  NB * NTPB; // Number of trajectories
    
    // Model Parameters
    float T = 1.0f;
    float rho = 0.5f; 
    float k = 0.5;
    float theta = 0.1;
    float S_0 = 1.0f;
    float K = S_0;
    float sigma = 0.3f;
    float r = 0.0f;
    int N = 100; // Time steps
    float dt = T / (float)N; // Standard time step
    float v_0 = 0.1f;

    // CPU Memory Allocation
    //float *PayCPU = (float*)malloc(n * sizeof(float));
    float *PayCPU = (float*)malloc(1 * sizeof(float)); // Only one float for the sum

    // GPU Memory Allocation
    float *PayGPU;
    //cudaMalloc((void**)&PayGPU, n * sizeof(float));
    cudaMalloc((void**)&PayGPU, 1 * sizeof(float));

    // RNG State Allocation
    curandStateXORWOW *state; 
    cudaMalloc((void**)&state, n * sizeof(curandStatePhilox4_32_10_t));

    // Random Seed (using system time or fixed value)
    unsigned long seed = 1234ULL; 

    // 1. Initialize RNG States
    // Arguments: State array, Global Seed
    init_curand_state_k<<<NB, NTPB>>>(state, seed);

    // Timing Setup
    float Tim;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 2. Launch Monte Carlo Kernel
    // Arguments: state, output array, and model parameters
    MC_euler<<<NB, NTPB>>>(rho,v_0,S_0, r, sigma, k, theta, dt, K, N,
            state, PayGPU, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Tim, start, stop);
    
    // 3. Copy Results back to Host
    //cudaMemcpy(PayCPU, PayGPU, n* sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(PayCPU, PayGPU, 1* sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = PayCPU[0]; // pricing estimation
    //float sum2 = 0.0f; // We would need to compute this separately if we want the confidence interval

    /* Reduction performed on the host
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += PayCPU[i];
        sum2 += PayCPU[i] * PayCPU[i];
    }
    
    sum = sum / n;
    sum2 = sum2 / n; 
    */

    printf("The estimated price is equal to %f\n", sum);
    //printf("error (95%% CI) = %f\n", 1.96 * sqrt((double)(1.0f / (n - 1)) * (n * sum2 - (n * sum * sum))) / sqrt((double)n));
    
    // Note: The Black-Scholes formula in your print statement is a simplified version for comparison
    printf("Execution time %f ms\n", Tim);

    // 4. Cleanup
    cudaFree(PayGPU);
    cudaFree(state);
    free(PayCPU);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}