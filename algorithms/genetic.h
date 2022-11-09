#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <vector>

using namespace std;

class Genetic{
    int populationSize;
    int maxIterations;
    int numberOfNodes;
    int maxBitSize;
    vector<vector<int>> paths;
    unordered_map<string, double> pathFitness;
    vector<int> lowerBound;
    vector<int> upperBound;
    vector<int> decisionNodes;
    vector<string> population;
    vector<int> bestSolution;
    double bestFitness;
    Graph* graph;
    
    public:
    Genetic(int numberOfNodes, vector<pair<int, int>> edges, int populationSize, int maxIterations){
        this->decisionNodes = {};
        this->numberOfNodes = numberOfNodes;
        this->populationSize = populationSize;
        this->maxIterations=maxIterations;
        graph = new Graph(numberOfNodes,edges);
        vector<int> outDegree = graph->getOutDegree();
        map<vector<int>, int> pFitness = graph->getPathFitness();
        calculateBounds(outDegree);
        generateChromosomes(pFitness, outDegree);
        initialisePopulation();
        //unitTesting();
    }

    //print
    void print(string &bestSolution){
        for(int i = 0; i < bestSolution.length(); i += maxBitSize){
            string currentNodeBinary = bestSolution.substr(i, maxBitSize);
            int currentNode = stoi(currentNodeBinary, nullptr, 2);
            cout << currentNode + 1 << " ";
        }
        cout << "," << pathFitness[bestSolution] << endl;
    }

    void calculateBounds(vector<int>& outDegree){
        for(int i = 0; i < numberOfNodes; i++){
            lowerBound.push_back(0);
            upperBound.push_back(outDegree[i]-1);
        }
    }

    bool isChromosomeInBounds(string chromosome){
        for(int i = 0; i < chromosome.length(); i += maxBitSize){
            string currentNodeBinary = chromosome.substr(i, maxBitSize);
            int currentNode = stoi(currentNodeBinary, nullptr, 2);
            if(currentNode < lowerBound[decisionNodes[i/maxBitSize]] || currentNode > upperBound[decisionNodes[i/maxBitSize]]){
                return false;
            }
        }
        return true;
    }

    //mutate
    string mutate(string chromosome){
        int trials = 100;
        while(trials--){
            int random = rand() % chromosome.length();
            string mutatedChromosome = chromosome;
            if(mutatedChromosome[random] == '0')
                mutatedChromosome[random] = '1';
            else
                mutatedChromosome[random] = '0';
            
            if(isChromosomeInBounds(mutatedChromosome))
                return mutatedChromosome;
        }
        return chromosome;
    }

    string substr(string chromosome, int start, int end){
        string sub = "";
        for(int i = start; i < end; i++){
            sub += chromosome[i];
        }
        return sub;
    }

    //crossover
    string crossover(string parent1, string parent2){
        assert(parent1.length() == parent2.length());
        assert(parent1.length() > 0);
        int trials = 100;
        while(trials--){
            int crossoverPoint = 1;
            string child1 = substr(parent1, 0, crossoverPoint) + substr(parent2, crossoverPoint, parent2.length());
            string child2 = substr(parent2, 0, crossoverPoint) + substr(parent1, crossoverPoint, parent1.length());
            if(isChromosomeInBounds(child1) && isChromosomeInBounds(child2)){
                if (pathFitness[child1] > pathFitness[child2])
                    return child1;
                return child2;
            }
        }
        return parent1;
    }

    void unitTesting(){
        vector<string> a;
        for(int i = 0; i < pow(2, 8); i++){
            string s = "";
            for(int j = 0; j < 8; j++){
                if(i & (1 << j))
                    s += '1';
                else
                    s += '0';
            }
            a.push_back(s);
        }
        for(auto i: a){
            if(isChromosomeInBounds(i)){
                if(pathFitness.find(i) != pathFitness.end()){
                    cout << i << endl;
                }
            }
        }
    }

    //initialise population
    void initialisePopulation(){
        int populationCount = 0;
        for(auto i: pathFitness){
            population.push_back(i.first);
            populationCount++;
            if(populationCount == populationSize){
                break;
            }
        }
        string bestSolution = "";
        double bestFitness = 0;
        for(int i = 0; i < maxIterations; i++){
            for(int j = 0; j < populationSize; j++){
                double randomNumber = (rand() % 100) / 100.00;
                if (randomNumber <= 0.2) {
                    population[j] = mutate(population[j]);                
                } else if (randomNumber <= 0.8) {
                    //current best solution not equal to this chromosome
                    for(int k = 0; k < populationSize; k++){
                        if((pathFitness[population[k]] > bestFitness) && (k != j)){
                            bestSolution = population[k];
                            bestFitness = pathFitness[population[k]];
                        }
                    }
                    population[j] = crossover(population[j], bestSolution);                                 
                }
            }
            bestFitness = 0;
            bestSolution = "";
        }

        for(int j = 0; j < populationSize; j++){
            if(pathFitness[population[j]] > bestFitness){
                bestSolution = population[j];
                bestFitness = pathFitness[population[j]];
            }
        }
        //print(bestSolution);
        this->bestFitness = bestFitness;
    }
    
    // recursive function to generate all possible chromosomes
    void generateChromosomes(map<vector<int>, int>& pFitness, vector<int>& outDegree){
        int maxBitSize = 1;
        for(int i = 0; i<numberOfNodes; i++){
            if(outDegree[i]>1){
                maxBitSize = max(maxBitSize, int(log2(outDegree[i]-1)+1));
                decisionNodes.push_back(i);
            }
        }
        this->maxBitSize = maxBitSize;
        for(auto i: pFitness){
            vector<int> path = i.first;
            string chromosome = "";
            for(int i = 0; i<numberOfNodes; i++){
                if(outDegree[i]>1){
                    int decimalValue = path[i];
                    string binaryValue = "";
                    while(decimalValue>0){
                        binaryValue += to_string(decimalValue%2);
                        decimalValue /= 2;
                    }
                    while(binaryValue.size()<maxBitSize){
                        binaryValue += "0";
                    }
                    reverse(binaryValue.begin(), binaryValue.end());
                    chromosome += binaryValue;
                }
            }
            pathFitness[chromosome] = i.second;
        }
    }

    int getBestFitness(){
        return this->bestFitness;
    }

};