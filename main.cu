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
    //Config GPU
    int NTPB = 512;
    int NB = 512;
    int n =  NB * NTPB; // Number of trajectories
    
    // Model Parameters
    float T = 1.0f;
    float rho = 0.5f; 
    float S_0 = 1.0f;
    float K = S_0;
    float r = 0.0f;
    float v_0 = 0.1f;


    // Simulation Parameters
    int N = 100; // Time steps
    float dt = T / (float)N; // Standard time step
    int N_coarse = 30; 
    float dt_coarse = T / (float)N_coarse;

    // CPU Memory Allocation
    float *PayCPU = (float*)malloc(2 * sizeof(float)); // mean and second moment

    // GPU Memory Allocation
    float *PayGPU;
    cudaMalloc((void**)&PayGPU, 2 * sizeof(float));

    // RNG State Allocation
    curandState *state; 
    cudaMalloc((void**)&state, n * sizeof(*state));
    // Random Seed (using system time or fixed value)
    unsigned long seed = 1234ULL; 
    
    // Header pour ton DataFrame
    printf("id,kappa,theta,sigma,feller_val,ms_euler,ms_almost,ms_exact,price_euler,price_almost,price_exact,err_euler,err_almost,err_exact\n");
    
    // 1. Initialize RNG States
    init_curand_state_k<<<NB, NTPB>>>(state, seed);

//Benchmarking Variables
float t_euler, t_almost, t_almost30, t_exact;
float price_euler, price_almost, price_almost30, price_exact;
float second_moment_euler, second_moment_almost, second_moment_almost30, second_moment_exact;
float error_euler, error_almost, error_almost30, error_exact;

//Timing Variables
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

size_t shared_mem_size = 2 * NTPB * sizeof(float);

FILE* file = fopen("params_feller.csv", "r");
    if (!file) {
        printf("Erreur : Impossible d'ouvrir le fichier de paramètres.\n");
        return 1;
    }

float k, theta, sigma;
int id = 0;
while (fscanf(file, "%f,%f,%f", &k, &theta, &sigma) == 3) {
                // Vérification de la condition de Feller
                float feller_lhs = 2.0f * k * theta;
                float feller_rhs = sigma * sigma;
                printf("Testing k=%.4f, theta=%.4f, sigma=%.4f, Feller LHS=%.4f, Feller RHS=%.4f\n", k, theta, sigma, feller_lhs, feller_rhs);



                    // --- 2. Mesure MC_euler ---
                    cudaMemset(PayGPU, 0, 2 * sizeof(float)); // Reset obligatoire
                    cudaEventRecord(start);
                    MC_euler<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&t_euler, start, stop);
                        // 3. Copy Results back to Host
                    cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);
                        
                    price_euler = PayCPU[0]/(float)n; // pricing estimation
                    float second_moment_euler = PayCPU[1]/(float)n;
                    float variance_euler = second_moment_euler - price_euler * price_euler;
                    error_euler = 1.96*sqrtf(variance_euler);

                        

                    // --- 3. Mesure MC_almost (Fine) ---
                    cudaMemset(PayGPU, 0, 2 * sizeof(float));
                    cudaEventRecord(start);
                    MC_almost<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&t_almost, start, stop);
                    cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);

                    price_almost = PayCPU[0]/(float)n; // pricing estimation
                    float second_moment_almost  = PayCPU[1]/(float)n;
                    float variance_almost = second_moment_almost - price_almost * price_almost;
                    error_almost = 1.96*sqrtf(variance_almost);

                    // --- 4. Mesure MC_almost (Coarse / "30") ---
                    cudaMemset(PayGPU, 0, 2 * sizeof(float));
                    cudaEventRecord(start);
                    MC_almost<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt_coarse, K, N_coarse, state, &PayGPU[0], &PayGPU[1], n);

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&t_almost30, start, stop);
                    cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);

                    price_almost30 = PayCPU[0]/(float)n; // pricing estimation
                    second_moment_almost30 = PayCPU[1]/(float)n; 
                    float variance_almost30 = second_moment_almost30 - price_almost30 * price_almost30;
                    error_almost30 = 1.96*sqrtf(variance_almost30);

                    // --- 5. Mesure MC_exact ---
                    cudaMemset(PayGPU, 0, 2 * sizeof(float));
                    cudaEventRecord(start);
                    MC_exact<<<NB, NTPB, shared_mem_size>>>(rho, v_0, S_0, r, sigma, k, theta, dt, K, N, state, &PayGPU[0], &PayGPU[1], n);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&t_exact, start, stop);
                    cudaMemcpy(PayCPU, PayGPU, 2* sizeof(float), cudaMemcpyDeviceToHost);

                    price_exact = PayCPU[0]/(float)n; // pricing estimation
                    second_moment_exact = PayCPU[1]/(float)n; 
                    float variance_exact = second_moment_exact - price_exact * price_exact;
                    error_exact = 1.96*sqrtf(variance_exact);

                    // --- 6. Nettoyage et Affichage ---
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);

                    //output CSV format
                    printf("%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f%,.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                           id, k, theta, sigma, feller_lhs - feller_rhs,
                           t_euler, t_almost, t_almost30, t_exact, price_euler, price_almost, price_almost30, price_exact, error_euler, error_almost, error_almost30);
                    
                    fflush(stdout); // Pour voir les résultats en temps réel
                    id++;
                }
            
     fclose(file);   
    



    // 4. Cleanup
    cudaFree(PayGPU);
    cudaFree(state);
    free(PayCPU);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}