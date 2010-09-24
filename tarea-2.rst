Tarea 2
=======
La tarea consiste en aproximar la siguiente integral en la GPU
usando la `regla de Simpson`_:

.. math::

    \int_{0}^{2^20} \sum_{k = 1}^{10000} \sin(2\pi kx)\,dx,

usando un intervalo de integración de largo `h = 1`.

.. _regla de Simpson: http://en.wikipedia.org/wiki/Simpson's_method

Qué hay que hacer
-----------------
* Escribir el programa que calcula la integral.

* Hacer experimentos para estudiar cómo influye el tamaño del bloque
  en la performance.

* Hacer experimentos para estudiar cómo influye el tamaño del vector
  en la performance.

* Hacer experimentos para estudiar cómo influye el valor de `M`
  (en otras palabras, el requerimiento de cómputo de la función `f`)
  en la performance.

Las referencias para programar en CUDA son la 
*CUDA C Best Practices Guide* y la
*CUDA*,
que pueden ser descargadas
en la `página de documentación de CUDA`_.

.. _página de documentación de CUDA: http://developer.nvidia.com/object/cuda_3_1_downloads.html

Entrega
-------
La fecha de entrega de la tarea 2 es el **lunes 11 de octubre**.
La entega consiste en:

* el código utilizado, empaquetado en un tarball llamado
  ``apellido-nombre-t2.tgz``, reemplazando con su nombre y su apellido.

* un informe de **máximo una página**, en el que se explique los resultados de
  los experimentos y las conclusiones sobre cómo influye cada decisión en la
  performance, entregando tanto la evidencia empírica como la explicación
  teórica.

