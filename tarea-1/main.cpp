#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "cpu-map.hpp"
#include "gpu-map.hpp"

// (1 << n) == (2 elevado a n)
static const unsigned N = 1 << 10;
static const char *data_file_name = "data_2e6";

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

    for (int i = 0; i < 10; ++i)
        std::cout << x1.at(i) << " ";
    std::cout << std::endl;

    // aplicar la función f a todos los elementos de x1 en la CPU
    std::cout << "x1[0] = " << x1.at(0) << std::endl;
    cpu_map<functor_g>(x1);
    std::cout << "f(x1[0]) = " << x1.at(0) << std::endl;

    // aplicar la función f a todos los elementos de x2 en la GPU
    std::cout << "x2[0] = " << x2.at(0) << std::endl;
    gpu_map('g', &x2[0], x2.size());
    std::cout << "f(x2[0]) = " << x2.at(0) << std::endl;
}

