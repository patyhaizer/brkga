#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// this file contains all definitions
#define BRKGA_pop_size 1000
#define ALPHABET_SIZE 20
#define CHROMOSSOME_SIZE 500
#define INSTANCE_SIZE 10

#define NUM_BLOCKS BRKGA_pop_size
#define NUM_THREADS 500

#define DEBUG_BRKGA(code) if (0) { code }

//const unsigned eliteSize = 0.25 * BRKGA_pop_size;  // number of elite items in the population
//const unsigned mutantsSize = 0.05 * BRKGA_pop_size;  // number of mutants introduced at each generation into the population
//const double rhoe = 0.70;  // probability that an offspring inherits the allele of its elite parent

#endif // CONFIGURATION_H