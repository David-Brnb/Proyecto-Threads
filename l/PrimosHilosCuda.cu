// =================================================================
//
// Archivo: SumaParesCuda.cpp
// Autor: David Bernabe
// Descripción: Este archivo implementa el conteo de números pares en un arreglo
//              Para compilar:
//		        !nvcc -arch=sm_75 -o app SumaParesCuda.cpp
//
// Copyright (c) 2024 por Tecnológico de Monterrey.
// Todos los derechos reservados. Puede reproducirse para cualquier propósito no comercial.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Definimos las constantes para la cantidad de datos y configuración de CUDA.
#define SIZE 5000000 // 5e6 elementos en el arreglo
#define THREADS 512  // Número de hilos por bloque
#define BLOCKS	32   // Número de bloques
#define N 10         // Número de repeticiones para calcular el tiempo promedio

/*
    Función del lado del dispositivo (GPU) que cuenta los números primos en el rango definido.
    - Se usa memoria compartida para almacenar los resultados parciales de los hilos.
    - Cada hilo itera sobre los índices asignados y verifica si son primos.
    - Se almacena la suma de los números primos encontrados en una variable auxiliar.
    - Se realiza una reducción dentro de cada bloque para acumular los resultados parciales.
    - Finalmente, se guarda el resultado por bloque en el arreglo `results`.
*/
__global__ void pares(long long *results) {
    __shared__ long long cache[THREADS]; // Memoria compartida para almacenar los resultados parciales

    // Identificamos la ubicación del hilo y el índice en la caché compartida
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int cacheIndex = threadIdx.x;

    // Variable auxiliar para almacenar la suma de los números primos encontrados
    long long aux = 0;
    while (tid < SIZE) {
        if (tid >= 2) {
            bool esPrimo = true;
            for (int j = 2; j * j <= tid; j++) {
                if (tid % j == 0) {
                    esPrimo = false;
                    break;
                }
            }
            if (esPrimo) aux += tid;
        }
        
        // Avanzamos el índice para que el hilo procese múltiples valores
        tid += blockDim.x * gridDim.x;
    }

    // Guardamos el resultado parcial en la memoria compartida
    cache[cacheIndex] = aux;
    __syncthreads();

    // Reducción dentro del bloque para acumular los resultados parciales
    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Almacenar el resultado del bloque en el arreglo global
    if (cacheIndex == 0) {
        results[blockIdx.x] = cache[cacheIndex];
    }
}

int main(int argc, char* argv[]) {
    long long *results, *d_r;
    int i;
    
    // Variables para medir el tiempo de ejecución
    high_resolution_clock::time_point start, end;
    double tiempoTotal;

    // Reservamos memoria en la CPU para los resultados
    results = new long long[BLOCKS];

    // Reservamos memoria en la GPU para los resultados de cada bloque
    cudaMalloc((void**) &d_r, BLOCKS * sizeof(long long));

    cout << "Iniciando ejecución...\n";
    tiempoTotal = 0;

    // Ejecutamos el kernel múltiples veces para obtener un tiempo promedio
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        pares<<<BLOCKS, THREADS>>>(d_r);

        end = high_resolution_clock::now();
        tiempoTotal += duration<double, std::milli>(end - start).count();
    }

    // Copiamos los resultados desde la GPU a la CPU
    cudaMemcpy(results, d_r, BLOCKS * sizeof(long long), cudaMemcpyDeviceToHost);

    // Sumamos los resultados de todos los bloques
    long long sumaTotal = 0;
    for (i = 0; i < BLOCKS; i++) {
        sumaTotal += results[i];
    }

    // Mostramos el resultado y el tiempo promedio de ejecución
    cout << "Resultado = " << sumaTotal << "\n";
    cout << "Tiempo promedio = " << fixed << setprecision(3) << (tiempoTotal / N) << " ms\n";

    // Liberamos la memoria asignada en la GPU y la CPU
    cudaFree(d_r);
    delete[] results;

    return 0;
}
