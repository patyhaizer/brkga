#ifndef INSTANCE_H
#define INSTANCE_H

class Instance
{
  private:
    // avoiding copy
    Instance(const Instance& other);
    Instance operator=(const Instance& other);
    Instance();

  public:
    char * m_alphabet;
    char ** m_instanceData;
    unsigned m_chromossomeSize, m_instanceSize, m_alphabetSize;

    Instance(char* alphabet, char** instanceData, unsigned chromossomeSize, unsigned instanceSize, unsigned alphabetSize);
    ~Instance();

  private:
    void printInstanceData();
    void printOneChromossome();
};

#endif // INSTANCE_H