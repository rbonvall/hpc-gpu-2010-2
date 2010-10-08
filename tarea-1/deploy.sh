#!/bin/sh

NAME=apellido-nombre-t1

mkdir $NAME
for i in Makefile *.cpp *.cu *.py *.hpp
do
    cp $i $NAME
done

tar zcf $NAME.tgz $NAME
mv $NAME.tgz ../_static/
rm -rf $NAME

