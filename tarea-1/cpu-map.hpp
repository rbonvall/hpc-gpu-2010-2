#pragma once
#include <vector>

template <typename FunctorType>
void cpu_map(std::vector<float>& x);

struct functor_f { float operator()(float); };
struct functor_g { float operator()(float); };

