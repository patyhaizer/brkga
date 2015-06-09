#include "BRKGA.h"
#include "BRKGAGPU.cu"

#include <time.h>
#include <iostream>

#define EXECUTION_TIME 3
#define DEBUG_BRKGA(code) if (0) { code }

BRKGA::BRKGA(const Instance * instance, const unsigned& chromossomeSize, const unsigned& popSize, const unsigned& eliteSize,
             const unsigned& mutantsSize, const double& rhoe)
    : m_instance(instance)
    , m_chromossomeSize(chromossomeSize)
    , m_popSize(popSize)
    , m_eliteSize(eliteSize)
    , m_mutantsSize(mutantsSize)
    , m_rhoe(rhoe)
{
    // Allocation populations:
    m_popPrevious = new Population(m_popSize, m_chromossomeSize);
    m_popCurrent = new Population(m_popSize, m_chromossomeSize);

}

BRKGA::~BRKGA() {
    delete m_popCurrent;
    delete m_popPrevious;
}

void
BRKGA::runBrkga() {
    // This function will call GPU functions in order to compute the algorithm
    // Allocate GPU variables

    // First call function to Initialize Current Population
    initializePopulation(m_popCurrent);
    time_t TStop = time(NULL), TStart = time(NULL);
    do {
        // Call GPU function to evaluate individuals
        // Order population individuals using thrust
        // Copy current to previous
        // Call GPU function to generate next generation
        TStop = time(NULL);
        DEBUG_BRKGA(std::cout<< TStop << std::endl;)
    } while ((TStop - TStart) < EXECUTION_TIME);
    // get best individual to return
}