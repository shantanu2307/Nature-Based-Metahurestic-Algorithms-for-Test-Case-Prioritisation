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

// Algorithms
#include "graph.h"
#include "genetic.h"
#include "grasshopper.h"
#include "cheetah.h"

using namespace std;


void compareAverageFitness(){
    int n, e;
    cin >> n >> e;
    vector<pair<int, int>> edges;
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({ u - 1, v - 1 });
    }

    // Comparing average fitness of GrassHopper and Genetic Algorithm 
    cout<<"N,"<<"IT,"<<"Average GrassHopper Fitness,"<<"Average Genetic Fitness"<<endl;
    for(int hyperparameters = 2; hyperparameters <= n/2; hyperparameters++){
        for(int maxIterations = 2; maxIterations <= n; maxIterations++){
            cout<<hyperparameters<<","<<maxIterations<<",";
            int trials = 1000;
            double averageGrasshopperFitness = 0;
            double averageGeneticFitness = 0;
            while(trials--){
                GrassHopper grasshopper(n, edges, hyperparameters, maxIterations, 1.19999999999999995559, 0.80000000000000004441, 1.00000000000000000000, 0.00010000000000000000);  
                averageGrasshopperFitness += grasshopper.getBestFitness();  
                Genetic genetic(n, edges, hyperparameters, maxIterations);
                averageGeneticFitness += genetic.getBestFitness();
            }
            cout<<averageGrasshopperFitness/1000.00<<","<<averageGeneticFitness/1000.00<<endl;
        }
    }
}

void getCountsWhereGrassHopperDominatesGenetic(){
    int n, e;
    cin >> n >> e;
    vector<pair<int, int>> edges;
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({ u - 1, v - 1 });
    }

    // Getting counts where GrassHopper is better than genetic algorithm
    cout<<"F,"<<"L,"<<"C,"<<"Count"<<endl;
    int hyperparameters = 6;
    int maxIterations = 12;
    for(double f = 0.1; f <= 2; f += 0.1){
        for(double l = 0.1; l <= 2; l += 0.1){
            for(double cmin = 0; cmin <= 2; cmin += 0.1){
                cout<<f<<","<<l<<","<<cmin<<",";
                int trials = 1000;
                int grasshopperWins = 0;
                while(trials--){
                    GrassHopper grasshopper(n, edges, hyperparameters, maxIterations, f, l, cmin, 0.00010000000000000000);  
                    int grasshopperFitness = grasshopper.getBestFitness();  
                    Genetic genetic(n, edges, hyperparameters, maxIterations);
                    int geneticFitness = genetic.getBestFitness();
                    grasshopperWins += (grasshopperFitness >= geneticFitness);
                }
                cout<<grasshopperWins<<endl;
            }
        }
    }

}

void printFitnessValues(){
    int n, e;
    cin >> n >> e;
    vector<pair<int, int>> edges;
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({ u - 1, v - 1 });
    }

    
    // Stack based Weight Algorithm
    Graph g(n, edges);
    // g.printNodeFitness();
    // g.printPathFitness();
}

int main()
{

    // ios_base::sync_with_stdio(false);
    // cin.tie(NULL);
    // cout.tie(NULL);

#ifndef ONLINE_JUDGE
    freopen("../input/student_enrollment.txt", "r", stdin);
    freopen("../output/output.txt", "w", stdout);
#endif

    srand(time(0)); 
    // printFitnessValues();
    // compareAverageFitness();
    // getCountsWhereGrassHopperDominatesGenetic();
    int n, e;
    cin >> n >> e;
    vector<pair<int, int>> edges;
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({ u - 1, v - 1 });
    }


    cout<<"N,"<<"IT,"<<"Average GrassHopper Fitness,"<<"Average Genetic Fitness,"<<"Average Cheetah Fitness"<<endl;
    for(int hyperparameters = 4; hyperparameters <=6; hyperparameters++){
        for(int maxIterations = 2; maxIterations <= 6; maxIterations++){
            cout<<hyperparameters<<","<<maxIterations<<",";
            int trials=1000;
            double averageGrasshopperFitness = 0;
            double averageGeneticFitness = 0;
            double averageCheetahFitness = 0;
            while(trials--){
                GrassHopper grasshopper(n, edges, hyperparameters, maxIterations, 1.19999999999999995559, 0.80000000000000004441, 1.00000000000000000000, 0.00010000000000000000);  
                averageGrasshopperFitness += grasshopper.getBestFitness();  
                Genetic genetic(n, edges, hyperparameters, maxIterations);
                averageGeneticFitness += genetic.getBestFitness();
                Cheetah cheetah(n, edges, hyperparameters,hyperparameters-1,maxIterations);
                averageCheetahFitness += cheetah.getBestFitness();
            }
            cout<<averageGrasshopperFitness/1000.00<<","<<averageGeneticFitness/1000.00<<","<<averageCheetahFitness/1000.00<<endl;
        }
    }


    
    return 0;
}