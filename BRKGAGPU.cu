#include "Population.h"
#include <iostream>

void initializePopulation(Population * pop) {
    std::cout << "initializing pop size: " << pop->m_size << std::endl;
}

void generateNextPopulation(Population * curr, Population * next) {
    // copy elite individuals
    // insert mutants
    // produce offspring
}