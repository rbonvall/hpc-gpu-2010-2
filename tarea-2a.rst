Tarea 2a
========
La tarea consiste en
estimar numéricamente
el valor de la integral definida:

.. math::

    \int_{-4}^{4} e^{-x/16} \cos^2\bigl(2\pi f(x) x\bigr)\,dx,

donde:

.. math::

    f(x) = \frac{1}{10} \sum_{i = 0}^{9} k\bigl(x - (-3.3 + 0.7i)\bigr)

y:

.. math::

    k(x) = 0.5 + \frac{1.5}{1 + 50x^2},

usando el `método de Simpson`_
con `n = 2^{22}` intervalos.

.. _método de Simpson: http://es.wikipedia.org/wiki/Regla_de_Simpson

