#include "configuration.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define tam_pop = 32

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

struct Instance {
    char m_alphabet[ALPHABET_SIZE];
    char m_instanceData[INSTANCE_SIZE][CHROMOSSOME_SIZE];
};

struct Individual {
    double m_chromossome[CHROMOSSOME_SIZE];
    double m_fitness;
};

struct Population {
    Individual m_chromossomes[BRKGA_pop_size][CHROMOSSOME_SIZE];
    unsigned m_size;
};

void loadFromFile(std::vector<string>& instanceData) {
    //COLOQUE AQUI O ARQUIVO QUE CONTÉM A INSTÂNCIA QUE PRETENDE EXECUTAR
    string instance_dir = "/Users/PatriciaHaizer/Documents/Instance/20caracteres/";
    //string output_dir = "/Users/PatriciaHaizer/Documents/Instance/20caracteres/Result/";

    //ARQUIVOS DE INSTÂNCIAS A SEREM UTILIZADOS
    std::vector<string> instance_name;
    instance_name.push_back("t20-10-500-1");

    // carregar o arquvio com as instâncias
    string tmp_name(instance_dir + instance_name[0] + ".txt");
    char * filename = new char[tmp_name.size() + 1];
    strcpy(filename, tmp_name.c_str());
    //string file_out = output_dir + "BRKGA_" + instance_name[0] + ".txt";

    std::vector<int> max, min, avg;

    ifstream fin(filename);

    int dummy;
    fin >> dummy;
    assert(dummy == INSTANCE_SIZE);
    fin >> dummy;
    assert(dummy == CHROMOSSOME_SIZE);

    string tmp = "";

    for(int i = 0; i < INSTANCE_SIZE; ++i) {
        fin >> tmp;
        instanceData.push_back(tmp);
        tmp = "";
    }
    fin.close();
    delete [] filename;
}

int main(int argc, char **argv) {
    std::vector<string> tmpInstanceData;
    loadFromFile(tmpInstanceData);
    assert(tmpInstanceData.size() == INSTANCE_SIZE);

    char h_alphabet[ALPHABET_SIZE +1];
    strcpy(h_alphabet, "0123456789abcdefghij");

    // convert string vector into char **
    char instanceData[INSTANCE_SIZE][CHROMOSSOME_SIZE];
    for (int i = 0; i < INSTANCE_SIZE ; ++i) {
        strcpy(instanceData[i], tmpInstanceData[i].c_str());
    }
    //const unsigned eliteSize = 0.25 * BRKGA_pop_size;  // number of elite items in the population
    //const unsigned mutantsSize = 0.05 * BRKGA_pop_size;  // number of mutants introduced at each generation into the population
    //const double rhoe = 0.70;  // probability that an offspring inherits the allele of its elite parent
    return 0;
}
