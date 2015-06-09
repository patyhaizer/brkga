
#include "Individual.h"

Individual::Individual(unsigned chromossomeSize)
    : m_fitness(0)
    , m_chromossomeSize(chromossomeSize)
{
    m_chromossome = new double[m_chromossomeSize]();
}
Individual::~Individual() {
    delete [] m_chromossome;
}