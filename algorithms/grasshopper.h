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


class GrassHopper {
    int numberOfGrasshoppers;
    int maxIterations;
    double f, l;
    double cmax, cmin;
    vector<vector<int>> paths;
    map<vector<int>, double> pathFitness;
    vector<vector<int>> observationWindow;
    vector<int> bestSolution;
    vector<int> lowerBound;
    vector<int> upperBound;
    int dimension;
    int bestFitness;
    Graph* graph;

public:
    GrassHopper(int numberOfNodes, vector<pair<int, int>> edges, int numberOfGrasshoppers, int maxIterations, double f, double l, double cmax, double cmin){
        this->numberOfGrasshoppers = numberOfGrasshoppers;
        this->maxIterations = maxIterations;
        this->f = f;
        this->l = l;
        this->cmax = cmax;
        this->cmin = cmin;        
        
        graph = new Graph(numberOfNodes, edges);
        vector<int> outDegree = graph -> getOutDegree();
        map<vector<int>, int> pFitness = graph -> getPathFitness();
        
        // Calculating Dimension
        int dimesnion = 0;
        for (int i = 0; i < numberOfNodes; i++) {
            if (outDegree[i] > 1) {
              dimesnion++;
            }
        }
        this->dimension = dimesnion;

        // Calculating assign fitness to each path (just decision nodes)
        for(auto i : pFitness) {
            vector<int> curr;
            for(int j = 0; j < numberOfNodes-1; j++){
                if(outDegree[j]>1){
                    curr.push_back(i.first[j]+1);
                }
            }
            paths.push_back(curr);
            pathFitness[curr] = i.second;
        }

        initiate();
    }

    void print(){
        vector<int> v = bestSolution;
        for (int i = 0; i < v.size(); i++) {
            cout << v[i] + 1 << " ";
        }
        cout << "," << pathFitness[v] << ",";
    }

    void check(vector<int> v){
        if (v[0] == 1 && v[1] == 1) {
            cout << numberOfGrasshoppers << " " << maxIterations << " " << f << " " << l << " " << cmax << " " << cmin << " " << dimension << endl;
        }
    }

    void calculateBounds(){
        for (int j = 0; j < dimension; j++) {
            int lb = INT_MAX;
            int ub = INT_MIN;
            for (auto i : this->pathFitness) {
                lb = min(lb, i.first[j]);
                ub = max(ub, i.first[j]);
            }
            lowerBound.push_back(lb);
            upperBound.push_back(ub);
        }
    }

    double calculateSocialInteraction(int r){
        return f * exp(-1 * (r / double(l))) - exp(-1 * r);
    }

    double calculateUnitVector(vector<int> a, vector<int> b, int direction){
        double sum = 0;
        for (int i = 0; i < dimension; i++) {
            sum += pow(a[i] - b[i], 2);
        }
        sum = sqrt(sum);
        return abs(a[direction] - b[direction]) / sum;
    }

    vector<int> normalizeDistance(int curr, double c){
        vector<int> normalizedDistance(dimension, 0);
        for (int i = 0; i < dimension; i++) {
            double distance = 0;
            for (int j = 0; j < numberOfGrasshoppers; j++) {
                if (observationWindow[j] == observationWindow[curr])
                    continue;
                double s = calculateSocialInteraction(abs(observationWindow[curr][i] - observationWindow[j][i]));
                double u = calculateUnitVector(observationWindow[curr], observationWindow[j], i);
                distance += c * ((upperBound[j] - lowerBound[j]) / 2.00) * s * u;
            }
            double normalized = c * distance + bestSolution[i];
            int roundedNormalized = round(normalized);
            int convertToLimits = roundedNormalized;
            normalizedDistance[i] = convertToLimits;
        }
        return normalizedDistance;
    }

    void initiate(){
        calculateBounds();
        unordered_set<int> covered;
        while (covered.size() < numberOfGrasshoppers) {
            int random = rand() % paths.size();
            if (covered.find(random) == covered.end()) {
                observationWindow.push_back(paths[random]);
                covered.insert(random);
            }
        }
        bestSolution = observationWindow[0];
        for (int i = 1; i < numberOfGrasshoppers; i++) {
            if (pathFitness[observationWindow[i]] > pathFitness[bestSolution]) {
                bestSolution = observationWindow[i];
            }
        }
        int currIteration = 1;
        while (currIteration < maxIterations) {
            double c = cmax - (currIteration * (cmax - cmin)) / double(maxIterations);
            vector<vector<int>> newObservationWindow;
            for (int i = 0; i < numberOfGrasshoppers; i++) {
                vector<int> normalizedDistance = normalizeDistance(i, c);
                newObservationWindow.push_back(normalizedDistance);
            }
            observationWindow = newObservationWindow;
            bestSolution = observationWindow[0];
            for (int i = 0; i < numberOfGrasshoppers; i++) {
                if (pathFitness[observationWindow[i]] > pathFitness[bestSolution]) {
                    bestSolution = observationWindow[i];
                }
            }
            currIteration++;
        }
        this->bestFitness = pathFitness[bestSolution];
    }

    int getBestFitness(){
        return this->bestFitness;
    }
};

