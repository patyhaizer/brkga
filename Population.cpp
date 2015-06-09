
#include "Population.h"

Population::Population(unsigned size, unsigned chromossomeSize)
    : m_size(size)
{
    m_chromossomes = new Individual*[size];
    for (int i = 0; i < m_size ; ++i) {
        m_chromossomes[i] = new Individual(chromossomeSize);
    }
}

Population::~Population() {
    for (int i = 0; i < m_size ; ++i) {
        delete m_chromossomes[i];
    }
    delete [] m_chromossomes;
}