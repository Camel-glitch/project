#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include "gamma.h"
#include "utils.h"
#include "MC_euler.h"
#include "MC_exact.h"
#include "MC_almost.h"

void testCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

int main(void) {
    int NTPB = 512;
    int NB = 512;
    int n = NB * NTPB;
    
    float T = 1.0f, rho = 0.5f, S_0 = 1.0f, K = S_0, r = 0.0f, v_0 = 0.1f;
    int N = 100;
    float dt = T / (float)N;
    int N_coarse = 30; 
    float dt_coarse = T / (float)N_coarse;

    float *PayCPU = (float*)malloc(2 * sizeof(float));
    float *PayGPU;
    testCUDA(cudaMalloc((void**)&PayGPU, 2 * sizeof(float)));
    curandState *state; 
    testCUDA(cudaMalloc((void**)&state, n * sizeof(*state)));
    
    unsigned long seed = 1234ULL; 
    
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    size_t shared_mem_size = 2 * NTPB * sizeof(float);

    FILE* file = fopen("params_feller.csv", "r");
    if (!file) {
        fprintf(stderr, "Erreur : Impossible d'ouvrir params_feller.csv\n");
        return 1;
    }

    // Header propre pour le CSV
    printf("id,kappa,theta,sigma,feller_margin,ms_e,ms_a,ms_a30,ms_ex,p_e,p_a,p_a30,p_ex,err_e,err_a,err_a30,err_ex\n");

    float k, theta, sigma;
    int id = 1;

    while (fscanf(file, "%f,%f,%f", &k, &theta, &sigma) == 3) {
        float feller_margin = (2.0f * k * theta) - (sigma * sigma);
        
        float t_e, t_a, t_a30, t_ex;
        float p_e, p_a, p_a30, p_ex;
        float err_e, err_a, err_a30, err_ex;

        // --- 1. MC_EULER ---
        cudaMemset(PayGPU, 0, 2 * sizeof(float));
        init_curand_state_k<<<NB, NTPB>>>(state, seed); // Reset RNG pour synchro
        cudaEventRecord(start);
        MC_euler<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_e, start, stop);
        cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);
        p_e = PayCPU[0]/n;
        err_e = 1.96f * sqrtf((PayCPU[1]/n - p_e*p_e)/n);

        // --- 2. MC_ALMOST (FINE) ---
        cudaMemset(PayGPU, 0, 2 * sizeof(float));
        init_curand_state_k<<<NB, NTPB>>>(state, seed); 
        cudaEventRecord(start);
        MC_almost<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_a, start, stop);
        cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);
        p_a = PayCPU[0]/n;
        err_a = 1.96f * sqrtf((PayCPU[1]/n - p_a*p_a)/n);

        // --- 3. MC_ALMOST (COARSE 30) ---
        cudaMemset(PayGPU, 0, 2 * sizeof(float));
        init_curand_state_k<<<NB, NTPB>>>(state, seed);
        cudaEventRecord(start);
        MC_almost<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt_coarse, K, N_coarse, state, &PayGPU[0], &PayGPU[1], n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_a30, start, stop);
        cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);
        p_a30 = PayCPU[0]/n;
        err_a30 = 1.96f * sqrtf((PayCPU[1]/n - p_a30*p_a30)/n);

        // --- 4. MC_EXACT ---
        cudaMemset(PayGPU, 0, 2 * sizeof(float));
        init_curand_state_k<<<NB, NTPB>>>(state, seed);
        cudaEventRecord(start);
        MC_exact<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_ex, start, stop);
        cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);
        p_ex = PayCPU[0]/n;
        err_ex = 1.96f * sqrtf((PayCPU[1]/n - p_ex*p_ex)/n);

        // Affichage CSV
        printf("%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               id, k, theta, sigma, feller_margin,
               t_e, t_a, t_a30, t_ex, 
               p_e, p_a, p_a30, p_ex, 
               err_e, err_a, err_a30, err_ex);
        
        fflush(stdout); 
        id++;
    }

    fclose(file);
    cudaFree(PayGPU);
    cudaFree(state);
    free(PayCPU);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}