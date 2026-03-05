cmake_minimum_required(VERSION 4.1)
project(ide_index C)

add_library(orange_avoider_ide STATIC
        orange_avoider.c
        orange_avoider.h)

target_include_directories(orange_avoider_ide PUBLIC
        ../.. .
        /home/ruben/CLionProjects/paparazzi/var/aircrafts/bebop_orange_avoid/nps
        ../../modules
        /home/ruben/CLionProjects/paparazzi/sw/airborne
)
