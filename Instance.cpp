#include "Instance.h"

#include <iostream>

#define DEBUG_INSTANCE(code) if (0) { code }

Instance::Instance(char* alphabet, char** instanceData, unsigned chromossomeSize, unsigned instanceSize, unsigned alphabetSize)
     : m_chromossomeSize(chromossomeSize)
     , m_instanceSize(instanceSize)
     , m_alphabetSize(alphabetSize)
{
     // allocating class pointers:
     m_alphabet = new char[m_alphabetSize + 1];
     memcpy(m_alphabet, alphabet, (sizeof(char) * (m_alphabetSize + 1)));
     m_instanceData = new char * [m_instanceSize]();
     for (int i = 0; i < m_instanceSize ; ++i) {
          m_instanceData[i] = new char[m_chromossomeSize + 1]();
          memcpy(m_instanceData[i], instanceData[i], (sizeof(char) * (m_chromossomeSize + 1)));
     }
     DEBUG_INSTANCE(printInstanceData();)

}

Instance::~Instance()
{
     // delete memory
     delete[] m_alphabet;
     for (int i = 0; i < m_instanceSize; ++i) {
         delete[] m_instanceData[i];
     }
     delete [] m_instanceData;
}

void
Instance::printInstanceData() {
     for (int i = 0; i < m_instanceSize; ++i) {
          std::cout << "Instance[" << i << "]: " << m_instanceData[i] << std::endl;
     }
}

void
Instance::printOneChromossome() {
     for (int i = 0; i < 1; ++i) {
          for (int j = 0; j < m_chromossomeSize; ++j) {
               std::cout << "Instance[" << i << "][" << j << "]: " << m_instanceData[i][j] << std::endl;
          }
     }
}

