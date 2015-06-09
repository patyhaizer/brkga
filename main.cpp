
#include "Instance.h"
#include "BRKGA.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <assert.h>

#define DEBUG_BRKGA(code) if (0) { code }

#define BRKGA_pop_size 1000
#define ALPHABET_SIZE 20

using namespace std;

unsigned g_chromosomeSize;    // Tamanho dos cromossomos
unsigned g_instanceSize;      // Tamanho da instancia

void loadFromFile(char* filename, std::vector<string>& instanceData) {
    ifstream fin(filename);

    fin >> g_instanceSize;
    fin >> g_chromosomeSize;

    DEBUG_BRKGA (std::cout << "Instance size: " << g_instanceSize << std::endl;)
    DEBUG_BRKGA (std::cout << "Chromossome size: " <<  g_chromosomeSize << std::endl;)
    string tmp = "";

    for(int i = 0; i < g_instanceSize; ++i) {
        fin >> tmp;
        instanceData.push_back(tmp);
        tmp = "";
    }
    fin.close();
}

int main(int argc, char **argv) {
    //COLOQUE AQUI O ARQUIVO QUE CONTÉM A INSTÂNCIA QUE PRETENDE EXECUTAR
    string instance_dir = "/Users/PatriciaHaizer/Documents/Instance/20caracteres/";
    string output_dir = "/Users/PatriciaHaizer/Documents/Instance/20caracteres/Result/";

    //ARQUIVOS DE INSTÂNCIAS A SEREM UTILIZADOS
    std::vector<string> instance_name;
    instance_name.push_back("t20-10-500-1");
    //instance_name.push_back("t20-10-500-2");
    //instance_name.push_back("t20-10-500-3");
    //instance_name.push_back("t20-10-500-4");
    //instance_name.push_back("t20-10-500-5");

    // carregar o arquvio com as instâncias
    string tmp_name(instance_dir + instance_name[0] + ".txt");
    char * filename = new char[tmp_name.size() + 1];
    strcpy(filename, tmp_name.c_str());
    string file_out = output_dir + "BRKGA_" + instance_name[0] + ".txt";

    std::vector<int> max, min, avg;
    std::vector<string> instanceData;
    loadFromFile(filename, instanceData);
    assert(instanceData.size() == g_instanceSize);

    char * h_alphabet = new char[ALPHABET_SIZE + 1];
    strcpy(h_alphabet, "0123456789abcdefghij");

    // convert string vector into char **
    char ** h_instanceData = new char * [g_instanceSize]();
    for (int i = 0; i < g_instanceSize ; ++i) {
        h_instanceData[i] = new char[g_chromosomeSize + 1]();
        strcpy(h_instanceData[i], instanceData[i].c_str());
    }

    Instance* inst = new Instance(h_alphabet, h_instanceData, g_chromosomeSize, g_instanceSize, ALPHABET_SIZE);
    const unsigned eliteSize = 0.25 * BRKGA_pop_size;  // number of elite items in the population
    const unsigned mutantsSize = 0.05 * BRKGA_pop_size;  // number of mutants introduced at each generation into the population
    const double rhoe = 0.70;  // probability that an offspring inherits the allele of its elite parent
    BRKGA* brkga = new BRKGA(inst, g_chromosomeSize, BRKGA_pop_size, eliteSize, mutantsSize, rhoe);
    brkga->runBrkga();

    // delete allocated memory
    for (int i = 0; i < g_instanceSize; ++i) {
        delete[] h_instanceData[i];
    }
    delete [] h_instanceData;
    delete [] filename;
    delete [] h_alphabet;
    delete inst;
    delete brkga;
    return 0;
}

