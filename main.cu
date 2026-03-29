#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include "gamma.h"
#include "utils.h"
#include "MC_euler.h"
#include "MC_exact.h"
#include "MC_almost.h"



// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))



int main(void) {
    int NTPB = 512;
    int NB = 512;
    int n =  NB * NTPB; // Number of trajectories
    
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
    float *PayCPU = (float*)malloc(2 * sizeof(float)); // Only one float for the sum

    // GPU Memory Allocation
    float *PayGPU;
    cudaMalloc((void**)&PayGPU, 2 * sizeof(float));

    // RNG State Allocation
    curandState *state; 
    cudaMalloc((void**)&state, n * sizeof(*state));
    // Random Seed (using system time or fixed value)
    unsigned long seed = 1234ULL; 

    // 1. Initialize RNG States
    // Arguments: State array, Global Seed
    init_curand_state_k<<<NB, NTPB>>>(state, seed);

float t_euler, t_almost, t_almost30, t_exact;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

size_t shared_mem_size = 2 * NTPB * sizeof(float);

int N_coarse = 30; 
float dt_coarse = T / (float)N_coarse;

// --- 2. Mesure MC_euler ---
cudaMemset(PayGPU, 0, 2 * sizeof(float)); // Reset obligatoire
cudaEventRecord(start);
MC_euler<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&t_euler, start, stop);
    // 3. Copy Results back to Host
cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);
    
float price_euler = PayCPU[0]/(float)n; // pricing estimation
float second_moment_euler = PayCPU[1]/(float)n; 

    

// --- 3. Mesure MC_almost (Fine) ---
cudaMemset(PayGPU, 0, 2 * sizeof(float));
cudaEventRecord(start);
MC_almost<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&t_almost, start, stop);
float price_almost = PayCPU[0]/(float)n; // pricing estimation
float second_moment_almost = PayCPU[1]/(float)n; 

// --- 4. Mesure MC_almost (Coarse / "30") ---
cudaMemset(PayGPU, 0, 2 * sizeof(float));
cudaEventRecord(start);
// On utilise N_coarse et dt_coarse ici
MC_almost<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt_coarse, K, N_coarse, state, &PayGPU[0], &PayGPU[1], n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&t_almost30, start, stop);
float price_almost30 = PayCPU[0]/(float)n; // pricing estimation
float second_moment_almost30 = PayCPU[1]/(float)n; 

// --- 5. Mesure MC_exact ---
cudaMemset(PayGPU, 0, 2 * sizeof(float));
cudaEventRecord(start);
MC_exact<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&t_exact, start, stop);
float price_exact = PayCPU[0]/(float)n; // pricing estimation
float second_moment_exact = PayCPU[1]/(float)n; 

// --- 6. Nettoyage et Affichage ---
cudaEventDestroy(start);
cudaEventDestroy(stop);

printf("Temps Euler: %.3f ms | Almost Fine: %.3f ms | Almost Coarse: %.3f ms | Exact: %.3f ms\n", 
        t_euler, t_almost, t_almost30, t_exact);



    

printf("The estimated price is equal to %f\n", price);
printf("error (95%% CI) = %f\n", 1.96 * sqrt((double)(1.0f / (n - 1)) * (n * second_moment - (n * price * price))) / sqrt((double)n));
    
    
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