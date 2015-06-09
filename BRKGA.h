#ifndef BRKGA_H
#define BRKGA_H

#include "Population.h"
#include "Instance.h"

class BRKGA
{
  private:
    // avoiding copy
    BRKGA(const BRKGA& other);
    BRKGA operator=(const BRKGA& other);
    BRKGA();

  public:
    Population * m_popPrevious, * m_popCurrent;
    const Instance * m_instance;
    // Hyperparameters:
    const unsigned m_chromossomeSize;   // number of genes in the chromosome
    const unsigned m_popSize;   // number of elements in the population
    const unsigned m_eliteSize;  // number of elite items in the population
    const unsigned m_mutantsSize;  // number of mutants introduced at each generation into the population
    const double m_rhoe;  // probability that an offspring inherits the allele of its elite parent

    BRKGA(const Instance * instance, const unsigned& chromossomeSize, const unsigned& popSize, const unsigned& eliteSize,
          const unsigned& mutantsSize, const double& rhoe);
    ~BRKGA();
    void runBrkga();
};

#endif // BRKGA_H