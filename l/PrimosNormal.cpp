// // =================================================================
// //
// // File: PrimosHilos.cpp
// // Author: David Bernabe
// // Description: This file implements the adding all prime numbers in a segment of numbers
// //		using C/C++ threads. To compile:
// //		g++ -o app PrimosHilos.cpp
// //
// // Copyright (c) 2024 by Tecnologico de Monterrey.
// // All Rights Reserved. May be reproduced for any non-commercial
// // purpose.
// //
// // =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

using namespace std;
using namespace std::chrono;

#define SIZE 5000000 // 5e9
#define N 10

/*
    Definimos getSum
    Este método nos permitirá sumar a todos
    de números primos desde 0 hasta size. 
    La complejidad temporal es O(n*√n)
*/
void getSum(long long &sum, int size) {
    sum=0;

    for (int i = 0; i < size; i++) {
        if(i<2){
            continue;
        }
        bool res=1;

        for(int j=2; j*j <= i; j++){
            if(i%j==0) res=0;
        }

        if(res) sum+=i;
    }

}

int main(int argc, char* argv[]) {
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point startTime, endTime;
    double timeElapsed;

    // Se comienza una pruebla que consta de 10 corridas
    // del método que linealmente nos dice la suma de 
    // los números primos.
    // En cada iteración, nosotros tomamos el tiempo 
    // de inicio y de término, sumando los tiempos
    // para al final imprimir el promedio.
    cout << "Starting...\n";
    long long totalSum;
    for (int j = 0; j < N; j++) {
        // Iniciamos el reloj
        startTime = high_resolution_clock::now();
        
        // Hacemos la prueba e imprimimos el resultado
        getSum(totalSum, SIZE);
        cout << totalSum << endl;

        // Terminamos el reloj y sumamos el sapso de tiempo;
        endTime = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(endTime - startTime).count();
    }

    // Imprimimos el promedio de tiempos
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}