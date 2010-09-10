#include "cpu-map.hpp"
#include <cmath>
#include <ctime>
#include <iostream>

/* Las funciones f y g aquí están implementadas como
 * objetos tipo función (functores). Esto tiene dos ventajas:
 * la función map puede ser parametrizada usando una plantilla,
 * y el compilador puede hacer que la llamada sea inline
 * (si se pasa un puntero a función, esto no es posible).
 */

float functor_f::operator()(float x) {
    float s = 0.0;
    for (int k = 1; k <= 10000; ++k) {
        s += std::sin(2 * M_PI * k * x);
    }
    return s;
}

float functor_g::operator()(float x) {
    return x * x;
}

/* Plantilla para una función que mapea los elementos del vector
 * usando cualquier functor que sea aplicable. */

template <typename FunctorType>
void cpu_map(std::vector<float>& x) {

    int n = x.size();
    FunctorType f;

    clock_t start = clock();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        x[i] = f(x[i]);
    }

    /* El ciclo for está paralelizado usando OpenMP.
     *
     * La directiva "#pragma omp parallel for" indica al compilador que las
     * iteraciones del ciclo pueden ejecutarse en paralelo. Para activar la
     * paralelización, hay que compilar con el flag -fopenmp.
     *
     * Por omisión serán creadas tantas hebras como cores haya disponibles.
     * Haga la prueba de ejecutar el ciclo con y sin paralelización, y compare
     * los tiempos de ejecución.
     */

    clock_t end = clock();
    double time = double(end - start) / CLOCKS_PER_SEC;

    std::cout << time << std::endl;
}

/* Instanciaciones explícitas de la plantilla */
template void cpu_map<functor_f>(std::vector<float>&);
template void cpu_map<functor_g>(std::vector<float>&);
