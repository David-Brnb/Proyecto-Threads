// =================================================================
//
// Archivo: PrimosHilos.cpp
// Autor: David Bernabe
// Descripción: Este archivo implementa la suma de todos los números primos en un segmento de números
//              utilizando hilos en C/C++. Para compilar:
//              g++ -o app PrimosHilos.cpp -pthread
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

#define SIZE 5000000 // Tamaño del segmento de números a evaluar
#define N 10 // Número de iteraciones para calcular el tiempo promedio

/*
    Función que calcula la suma de todos los números primos en un rango de 0 hasta 'size'.
    - Se recorre el rango de números verificando si cada uno es primo.
    - Se usa un método de verificación con divisores hasta la raíz cuadrada del número.
    - Si el número es primo, se acumula en la variable 'sum'.
    - Complejidad temporal: O(n * √n).
*/
void getSum(long long &sum, int size) {
    sum = 0;

    for (int i = 0; i < size; i++) {
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
    double timeElapsed = 0;

    // Se ejecuta el cálculo de la suma de primos en 'N' iteraciones para calcular un tiempo promedio
    cout << "Iniciando ejecución...\n";
    long long totalSum;
    
    for (int j = 0; j < N; j++) {
        // Inicio del conteo de tiempo
        startTime = high_resolution_clock::now();
        
        // Se ejecuta la función para calcular la suma de primos en el rango definido
        getSum(totalSum, SIZE);
        cout << "Suma total de primos: " << totalSum << endl;

        // Fin del conteo de tiempo y acumulación de tiempos
        endTime = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(endTime - startTime).count();
    }

    // Se imprime el tiempo promedio de ejecución del método
    cout << "Tiempo promedio = " << fixed << setprecision(3) 
         << (timeElapsed / N) << " ms\n";

    return 0;
}
