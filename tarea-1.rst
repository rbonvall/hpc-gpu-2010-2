Tarea 1
=======
La tarea consiste en aplicar una función
a todos los elementos de un arreglo.
(«mapear el arreglo»).

La función es la siguiente:

.. math::

    f(x) = \sum_{k = 1}^M \sin(2\pi kx).

Para evaluar la performance del programa,
usaremos varios valores de `M`.

Archivos de la tarea
--------------------
El `código base`_ contiene los siguientes archivos:

* ``main.cpp``: carga los datos del archivo a un vector,
  llama las funciones para mapear el arreglo
  en la CPU y en la GPU,
  y calcula la norma de la diferencia entre ambos resultados
  (para verificar que entregan el mismo resultado).

* ``cpu-map.cpp``:
  mapeo del vector implementado en C++ para correr en la CPU.
  Está paralelizado usando OpenMP.

* ``gpu-map.cu``:
  mapeo del arreglo en CUDA para correr en la GPU.
  En este archivo usted debe escribir la función
  que reserve la memoria global,
  cargue los datos,
  ejecute el kernel,
  mida el tiempo
  y copie el resultado de vuelta.

* ``gpu-map.hpp``.
* ``cpu-map.hpp``.
* ``Makefile``: mírenlo para ver cómo se compila con ``nvcc``.

* ``create_data_file.py``: script para crear el archivo de datos.
  Requiere tener Python y NumPy instalados.
  Útil por si quieren generar su propio archivo de datos.

.. _código base: _static/apellido-nombre-t1.tgz

El programa lee los datos de un archivo llamado ``data_2^26``
que tiene `2^{26}` números ``float`` de 4 bytes cada uno.
El programa no necesariamente los carga todos:
la constante ``N`` indica cuántos de los datos serán leídos en el arreglo.
Pueden generar su propio archivo de datos usando el script,
o usar el que tengo en mi cuenta del servidor ``gp01``,
en ``/home/utfsm/rbonvall/data_2^26``.

Qué hay que hacer
-----------------
* Implementar la función ``gpu_map`` que reserva la memoria,
  copia los datos, ejecuta el kernel y toma el tiempo de ejecución.

* Hacer experimentos para estudiar cómo influye el tamaño del bloque
  en la performance.

* Hacer experimentos para estudiar cómo influye el tamaño del vector
  en la performance.

* Hacer experimentos para estudiar cómo influye el valor de `M`
  (en otras palabras, el requerimiento de cómputo de la función `f`)
  en la performance.

Para saber cómo medir el tiempo de ejecución de un kernel CUDA,
busquen en la *CUDA C Best Practices Guide*
en la `página de documentación de CUDA`_.

.. _página de documentación de CUDA: http://developer.nvidia.com/object/cuda_3_1_downloads.html

Entrega
-------
La fecha de entrega de la tarea 1 es el **lunes 27 de septiembre**.
La entega consiste en:

* el código utilizado, empaquetado en un tarball llamado
  ``apellido-nombre-t1.tgz``, reemplazando con su nombre y su apellido;
  incluya un archivo ``README`` que indique brevemente
  qué modificaciones hizo al código original
  (haga todas las que estime conveniente).

* un informe de **máximo una página**, en el que se explique los resultados de
  los experimentos y las conclusiones sobre cómo influye cada decisión en la
  performance, entregando tanto la evidencia empírica como la explicación
  teórica.

