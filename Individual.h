#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

class Individual {
  private:
    Individual();
    Individual(const Individual& other);
    Individual operator=(const Individual& other);
  public:
    explicit Individual(unsigned chromossomeSize);
    ~Individual();
    double * m_chromossome;
    double m_fitness;
    unsigned m_chromossomeSize;
};

#endif // INDIVIDUAL_H