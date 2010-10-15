Cómo dejar trabajos encolados en el clúster
===========================================

Hasta ahora hemos estado trabajando
directamente en los servidores de GPU
vía conexión SSH.

Ésta no es la mejor manera de trabajar,
por dos motivos:

* es incómodo trabajar remotamente,
  ya que la conexión puede cortarse o ponerse lenta,
  y hay que quedarse esperando que los programas terminen; y
* puede haber varias personas corriendo procesos al mismo tiempo,
  por lo que tienen que turnarse los recursos.

La manera usual de trabajar en un clúster
es usando un sistema de trabajos por lotes.
El que usa nuestro clúster se llama PBS_.

.. _PBS: http://en.wikipedia.org/wiki/Portable_Batch_System

Para mandar un proceso a la cola de ejecución,
hay que hacer dos cosas:

1. Crear un *wrapper script* que tenga los comandos
   para ejecutar el proceso.
   Puede ser algo tan simple como:

   .. code-block:: bash

      #!/bin/bash
      /home/utfsm/rbonvall/hola-mundo

2. Enviar el script a la cola usando el programa ``qsub``,
   e indicando a qué cola se desea enviar el proceso
   (en este caso, la cola de los nodos con GPU):

   .. code-block:: console

     rbonvall@ui:~$ qsub -q gpu wrapper.sh
     248829.ce.labmc.inf.utfsm.cl

   La salida del programa muestra el identificador del trabajo.

La cola también se puede indicar mediante una directiva ``#PBS``
en el script:
 
.. code-block:: bash

  #!/bin/bash
  #PBS -q gpu
  /home/utfsm/rbonvall/hola-mundo

Una vez que el trabajo haya sido completado,
aparecerán dos archivos que tienen las salidas
estándar y de error de los programas ejecutados:

.. code-block:: console

  rbonvall@ui:~$ ls
  wrapper.sh.e248829
  wrapper.sh.o248829
  rbonvall@ui:~$ cat wrapper.sh.o248829
  Hola Mundo

Uno puede ver el estado de la cola usando el programa ``qstat``:

.. code-block:: console

  rbonvall@ui:~$ qstat
  Job id     Name             User            Time Use S Queue
  ---------- ---------------- --------------- -------- - -----
  248586.ce  Launcher         marat           09:19:33 R utfsm          
  248588.ce  dy-pdf           nemcik          41:22:16 R otros          
  248666.ce  dy-pdf           nemcik          20:48:25 R otros          
  248714.ce  ...en_collimator cmorgoth        11:25:45 R utfsm          
  248718.ce  ..._new_diameter cmorgoth        11:21:59 R utfsm          
  248753.ce  dy-pdf           nemcik          10:29:28 R otros          
  248762.ce  dy-f2p           nemcik          00:00:00 R otros          
  248773.ce  dy-f2p           nemcik          00:00:00 R otros          
  248816.ce  dy-f2p           nemcik          00:00:00 R otros          
  248818.ce  dy-f2p           nemcik          00:22:02 R otros          
  248819.ce  ...e-pcs-1-cpu-4 tsc06                  0 Q otros          
  248829.ce  wrapper.sh       rbonvall        00:00:04 C gpu            

Se puede aprovechar el script para que haga trabajo adicional
además del programa.  Por ejemplo:

.. code-block:: bash

  #!/bin/bash
  #PBS -q gpu
  for block_size in $(seq 32 32 512)
  do
      echo BLOCK SIZE: $block_size
      time /home/utfsm/rbonvall/programa $block_size
  done

