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
    freopen("../input/student_enrollment.txt", "r", stdin);
    freopen("../output/output.txt", "w", stdout);
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
    Graph g(n, edges);
    g.printNodeFitness();
    // int maxIterations=6;
    // cout<<"Sample,";
    // for(int i=0;i<1000;i++){
    //     cout<<"trial "<<i<<",";
    // }
    // cout<<endl;
    // for(int sample=4;sample<=6;sample++){
    //     cout<<sample<<",";
    //     for(int trial=1;trial<=1000;trial++){
    //         // Genetic gen(n, edges, sample, maxIterations);
    //         GrassHopper gen(n, edges, sample, maxIterations, 1.19999999999999995559, 0.80000000000000004441, 1.00000000000000000000, 0.00010000000000000000); 
    //         // Cheetah gen(n, edges, sample, sample-1, maxIterations);
    //         int bestFitness=gen.getBestFitness();
    //         cout<<bestFitness<<",";
    //     }
    //     cout<<endl;        
    // }



    
    return 0;
}