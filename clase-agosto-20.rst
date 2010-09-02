Clase 20 de agosto
==================
* Contenido:

  * introducción a la computación en procesadores masivos (`ver diapos`_)

* Ya solicité acceso al servidor con las GPU.
  Debería haberles llegado un mail con las indicaciones para entrar.
* Tarea: leer el artículo `GPU Computing <http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=04490127>`_
  (deben estar dentro de la UTFSM para poder descargarlo).

.. _ver diapos: http://docs.google.com/viewer?url=http%3A%2F%2Fstanford-cs193g-sp2010.googlecode.com%2Fsvn%2Ftrunk%2Flectures%2Flecture_1%2Fintroduction_to_massively_parallel_computing.pdf

Computación en procesadores masivos
-----------------------------------

Conceptos importantes:

* performance: capacidad de realizar instrucciones individuales en un tiempo dado.
* throughput: capacidad de realizar una tarea completa en un tiempo dado.

En general, para cualquier problema computacional
lo deseable es tener un buen troughput:
resolver el problema completo en poco tiempo.

El enfoque tradicional del diseño de procesadores era mejorar la performance
para lograr mejor througput. Esto se lograba metiendo cada vez más transistores
dentro del chip. Limitaciones físicas (tamaño del transistor, consumo de energía)
hacen que esta tendencia no pueda seguir indefinidamente.

La arquitectura de las GPU apunta a mejorar el throughput a través del paralelismo.
En vez de tener un procesador rápido y sofisticado,
cada chip tiene muchos procesadores que trabajan simultáneamente.
Muchas tareas computacionalmente costosas
se adaptan muy bien a esta arquitectura.

