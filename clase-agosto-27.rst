Clase 27 de agosto
==================
* Contenido:

  * conceptos de computación paralela,
  * introducción a CUDA (`ver diapos`_)
    
.. _ver diapos: http://docs.google.com/viewer?url=http%3A%2F%2Fstanford-cs193g-sp2010.googlecode.com%2Fsvn%2Ftrunk%2Flectures%2Flecture_2%2Fgpu_history_and_cuda_programming_basics.pdf

Conceptos importantes de computación paralela
---------------------------------------------
Las arquitecturas paralelas se clasifican según la `taxonomía de Flynn`_.
Las GPU son parecidas al modelo SIMD de esta taxonomía,
con algunas diferencias.

Paralelamente (*no pun intended*) a la clasificación de arquitecturas,
las estrategias para resolver problemas en paralelo suelen clasificarse en:

* `paralelismo de tareas`_: cada procesador realiza una tarea propia, y
* `paralelismo de datos`_: cada procesador realiza la misma tarea,
  pero sobre su propio conjunto de datos.

Ambas estrategias no son excluyentes. Por ejemplo,
los nodos de un clúster pueden estar ejecutando tareas diferentes,
y el procesador multicore de cada nodo puede estar haciendo paralelismo de datos.

El speedup_ de un programa paralelo
es la razón entre el tiempo `t_1` que tarda la versión serial del programa
y el tiempo `t_p` que tarda la versión paralela del programa.

En general, al paralelizar un programa usando `p` procesadores
no puede esperarse que el tiempo de ejecución sea `t_1/p` (speedup `p`).
Las razones que impiden este escalamiento lineal son:

* no todas las secciones del programa son paralelizables,
* se pierde tiempo en sincronizar los procesadores
  para coordinar el acceso a recursos compartidos, y
* se pierde tiempo al haber comunicación entre los procesadores.

Por supuesto, los problemas que no tienen secciones seriales,
no necesitan compartir recursos y no requieren comunicación
son los más paltosos de paralelizar.
Estos problemas se llaman `problemas ridículamente paralelos`_.

La `ley de Ahmdal`_ indica cuál es el máximo speedup esperable
en función de la proporción serial de un programa:

.. raw:: html

    <object data="http://upload.wikimedia.org/wikipedia/commons/e/ea/AmdahlsLaw.svg" type="image/svg+xml" width="324" height="243"></object>

Es importante entender cualitativamente esta ley,
pues permite reconocer cuándo no es posible mejorar la performance
metiendo más procesadores.

La ley de Ahmdal supone que la carga de trabajo es fija.
La `ley de Gustafson`_ indica el máximo speedup esperable
pensando que el paralelismo permite aumentar la carga de trabajo
que puede ser realizada en un período de tiempo fijo.

.. _speedup: http://en.wikipedia.org/wiki/Speedup
.. _taxonomía de Flynn: http://en.wikipedia.org/wiki/Flynn%27s_taxonomy
.. _ley de Ahmdal: http://en.wikipedia.org/wiki/Ahmdal%27s_Law
.. _ley de Gustafson: http://en.wikipedia.org/wiki/Gustafson%27s_law
.. _problemas ridículamente paralelos: http://en.wikipedia.org/wiki/Embarrassingly_parallel_problem
.. _paralelismo de tareas: http://en.wikipedia.org/wiki/Task_parallelism
.. _paralelismo de datos: http://en.wikipedia.org/wiki/Data_parallelism

Introducción a CUDA
-------------------
Las GPU son arreglos de multiprocesadores,
cada uno de los cuales tiene

* varios cores,
  que ejecutan el mismo programa concurrentemente,
* memoria compartida, y
* un mecanismo de sincronización.

Los servidores de GPU del LabMC tienen dos tarjetas,
cada uno con 30 multiprocesadores de 8 cores.

Este hardware se programa usando el entorno de programación CUDA,
que incluye extensiones al lenguaje C y una biblioteca de tiempo de ejecución
(runtime library).
Los programas CUDA deben ser compilados usando ``nvcc``.

La principal abstacción es el kernel:
una función que es ejecutada en paralelo
por muchas hebras.
La sintaxis para definir un kernel global es::

    __global__ void
    nombre(tipo_1 parametro_1, ..., tipo_n parametro_n) {
        /* codigo */
    }

La sintaxis para ejecutar un kernel global es::

    nombre<<<numero_bloques, numero_hebras_por_bloque>>>(x1, ..., xn);

Las hebras son creadas en bloques.
Los bloques pueden ejecutarse en cualquier orden,
por lo que deben ser independientes entre ellos.
Las hebras dentro de un mismo bloque
son ejecutadas en paralelo
dentro de un mismo multiprocesador.

Cada bloque y cada hebra tienen un identificador
que les permite saber qué hacer o sobre qué datos operar.

La documentación principal para programar en CUDA
es la `CUDA Programming Guide`_.

.. _CUDA Programming Guide: http://developer.nvidia.com/object/cuda_3_1_downloads.html

