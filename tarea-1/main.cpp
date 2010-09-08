#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "cpu-map.hpp"
#include "gpu-map.hpp"

// (1 << n) == (2 elevado a n)
static const unsigned N = 1 << 20;
static const char *data_file_name = "data_2^20";

int main(int argc, char *argv[]) {

    // Los vectores en C++ son escencialmente arreglos en C con una interfaz
    // más conveniente. Donde sea que se pueda usar arreglos, se puede usar
    // vectores. Cuando se necesita la dirección de memoria de un arreglo, hay
    // que usar la dirección del primer elemento del vector: &x[0].  x por sí
    // solo no sirve ya que es un objeto que encapsula al verdadero arreglo.

    std::vector<float> x1;
    x1.reserve(N);
    x1.resize(N);

    // copiar datos del archivo de entrada al vector
    std::ifstream input_file(data_file_name, std::ios::binary | std::ios::in);
    if (!input_file) {
        std::cerr << "No se pudo abrir el archivo " << data_file_name << std::endl;
        std::exit(-1);
    }
    input_file.read((char *) &x1[0], N * sizeof(float));
    input_file.close();

    // crear una copia del vector, de modo de tener
    // uno para mapear en la GPU y otro en la CPU
    std::vector<float> x2(x1);

    // aplicar la función f a todos los elementos de x1 en la CPU
    cpu_map<functor_g>(x1);

    // aplicar la función f a todos los elementos de x2 en la GPU
    gpu_map('g', &x2[0], x2.size());

    // verificar que los resultados son prácticamente iguales
    float squared_diff_norm = 0.0;
#   define SQUARED(x) ((x) * (x))
#   pragma omp parallel reduction(+: squared_diff_norm)
    for (unsigned i = 0; i < N; ++i)
        squared_diff_norm += SQUARED(x1[i] - x2[i]);
    std::cout << "Norm of the difference: " << std::sqrt(squared_diff_norm) << std::endl;
}

