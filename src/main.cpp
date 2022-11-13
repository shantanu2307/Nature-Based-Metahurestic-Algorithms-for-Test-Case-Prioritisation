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

int main()
{

#ifndef ONLINE_JUDGE
    freopen("../input/online_banking.txt", "r", stdin);
    freopen("../output/max_fitness.txt", "w", stdout);
#endif

    srand(time(0)); 
    int n, e;
    cin >> n >> e;
    vector<pair<int, int>> edges;
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({ u - 1, v - 1 });
    }

    int maxIterations=6;
    for(int sample=4;sample<=6;sample++){
        Genetic gen(n,edges, sample, maxIterations);
        GrassHopper grasshopper(n, edges, sample, maxIterations, 1.19999999999999995559, 0.80000000000000004441, 1.00000000000000000000, 0.00010000000000000000);
        Cheetah cheetah(n,edges, sample, sample-1, maxIterations);
        int geneticMaxFitness = gen.getBestFitness();
        int grasshopperMaxFitness = grasshopper.getBestFitness();
        int cheetahMaxFitness = cheetah.getBestFitness();
        cout<<geneticMaxFitness<<","<<grasshopperMaxFitness<<","<<cheetahMaxFitness<<endl;
    }



    
    return 0;
}