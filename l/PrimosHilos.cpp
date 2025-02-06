// =================================================================
//
// Archivo: SumaPares.cpp
// Autor: David Bernabe
// Descripción: Este archivo implementa el conteo de números primos en un arreglo
//              utilizando hilos en C/C++. Para compilar:
//              g++ -o app SumaPares.cpp -pthread
//
// Copyright (c) 2024 por Tecnológico de Monterrey.
// Todos los derechos reservados. Puede reproducirse para cualquier propósito no comercial.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::chrono;

#define SIZE   5000000 // Tamaño del arreglo a evaluar
#define THREADS std::thread::hardware_concurrency() // Número de hilos basados en el hardware
#define N 10 // Número de iteraciones para promediar el tiempo

/*
    Función que calcula la suma de los números primos en un rango determinado.
    - Se recorre el rango desde 'start' hasta 'end'.
    - Se verifica si un número es primo evaluando sus divisores.
    - Si es primo, se acumula en la variable 'sum'.
    - Complejidad temporal: O(n * √n).
*/
void getSum(int start, int end, long long &sum) {
    sum = 0;

    for (int i = start; i < end; i++) {
        if (i < 2) continue;
        
        bool esPrimo = true;
        for (int j = 2; j * j <= i; j++) {
            if (i % j == 0) {
                esPrimo = false;
                break;
            }
        }
        
        if (esPrimo) sum += i;
    }
}

int main(int argc, char* argv[]) {
    // Variables para medir el tiempo de ejecución
    high_resolution_clock::time_point startTime, endTime;
    double timeElapsed;

    // Variables para el manejo de hilos
    int end, blockSize;
    thread threads[THREADS];
    long long result[THREADS];
    blockSize = SIZE / THREADS; // Tamaño del bloque asignado a cada hilo

    // Se inicia la ejecución con múltiples hilos en paralelo
    cout << "Iniciando ejecución...\n";
    timeElapsed = 0;
    long long totalSum;
    
    for (int j = 0; j < N; j++) {
        // Inicio del conteo de tiempo
        startTime = high_resolution_clock::now();

        // Creación y ejecución de los hilos asignándoles su respectivo rango
        for (int i = 0; i < THREADS; i++) {
            end = (i != (THREADS - 1)) ? (i + 1) * blockSize : SIZE;
            threads[i] = thread(getSum, (i * blockSize), end, std::ref(result[i]));
        }

        // Esperamos a que todos los hilos finalicen su ejecución
        // y acumulamos los resultados parciales en 'totalSum'.
        totalSum = 0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            totalSum += result[i];
        }
        cout << "Suma total de primos: " << totalSum << endl;

        // Fin del conteo de tiempo y acumulación de tiempos
        endTime = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(endTime - startTime).count();
    }

    // Se imprime el tiempo promedio de ejecución del método con hilos
    cout << "Tiempo promedio = " << fixed << setprecision(3) 
         << (timeElapsed / N) << " ms\n";

    return 0;
}
