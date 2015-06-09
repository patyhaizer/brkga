#ifndef POPULATION_H
#define POPULATION_H

#include "Individual.h"

class Population {
  private:
    Population();
    Population(const Population& other);
    Population operator=(const Population& other);
  public:
    Population(unsigned size, unsigned chromossomeSize);
    ~Population();
    Individual ** m_chromossomes;
    unsigned m_size;
};

#endif