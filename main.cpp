#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

class Graph {
    int numberOfNodes;
    vector<vector<int>> adj;
    vector<int> inDegree;
    vector<int> outDegree;
    vector<int> informationGain;
    unordered_map<int, int> stackBasedWeight;
    unordered_map<int, int> fitnessValue;
    vector<vector<int>> paths;
    vector<int> reverseHeight;
    unordered_map<int, int> depth;
    map<vector<int>, int> pathFitness;

public:
    // Function to input the graph
    Graph(int numberOfNodes, vector<pair<int, int>> edges) {
        this->numberOfNodes = numberOfNodes;
        paths = {};
        adj = vector<vector<int>>(numberOfNodes);
        inDegree = vector<int>(numberOfNodes, 0);
        outDegree = vector<int>(numberOfNodes, 0);
        reverseHeight.resize(numberOfNodes);
        for (auto edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back(v);
            inDegree[v]++;
            outDegree[u]++;
        }
        calculateTotalFitness();
        traverseAllPaths();
    }

    // Function to assign height from end node to each node
    void assignReverseHeight(int curr, int parent, vector<bool>& visited){
        visited[curr] = true;
        reverseHeight[curr] = 0;
        for (int node : adj[curr]) {
            if ((node != parent) && !visited[node]) {
                assignReverseHeight(node, curr, visited);
                reverseHeight[curr] = max(reverseHeight[curr], reverseHeight[node] + 1);
            }
        }
    }

    // Function to calculate information gain (Indegree * Outdegree)
    void calculateInformationGain(){
        for (int i = 0; i < numberOfNodes; i++) {
            informationGain.push_back(inDegree[i] * outDegree[i]);
        }
    }

    void dfs(int node, int level, vector<bool>& vis, int maxDepth, vector<unordered_set<int>>& distinctLevels){
        distinctLevels[node].insert(level);
        vis[node] = true;
        for (int child : adj[node]) {
            if (!vis[child]) {
                dfs(child, level + 1, vis, maxDepth, distinctLevels);
            } else {
                distinctLevels[child].insert(level + 1);
            }
        }
    }

    // Function to calculate the stack based weight of a graph
    void calculateStackBasedWeight(){
        // weight of current node = maximum height of any node - minimum depth of current node
        int maxDepth = 0;
        for (int i = 0; i < numberOfNodes; i++) {
            maxDepth = max(maxDepth, reverseHeight[i]);
        }
        vector<bool> vis(numberOfNodes, false);
        vector<unordered_set<int>> distinctLevels(numberOfNodes);
        dfs(0, 0, vis, maxDepth, distinctLevels);
        for (int i = 0; i < numberOfNodes; i++) {
            for (auto j : distinctLevels[i]) {
                stackBasedWeight[i] += (maxDepth - j);
            }
        }
    }

    // Function to calculate fitness value of a node
    void calculateTotalFitness(){
        vector<bool> visited(numberOfNodes, 0);
        assignReverseHeight(0, -1, visited);
        calculateInformationGain();
        calculateStackBasedWeight();
        for (int i = 0; i < numberOfNodes; i++) {
            fitnessValue[i] = stackBasedWeight[i] + informationGain[i];
        }
    }

    // Function to generate all test paths
    void generateTestPaths(int i, vector<int> current){
        if (i == numberOfNodes - 1) {
            this->paths.push_back(current);
            return;
        }
        for (int j = 0; j < outDegree[i]; j++) {
            current.push_back(j);
            generateTestPaths(i + 1, current);
            current.pop_back();
        }
    }

    // Function to calculate fitness value of a path
    int traverseCurrentPath(int curr, vector<unordered_set<int>>& visitedEdge, vector<int> path){
        // newNode = adj[curr][path[curr]]
        // if there is only 1 outgoing edge dont mark it as visited else we will not be able to traverse cycles
        // if newNode traveresed, then go to max(reverseHeight(adj[curr][j])) where j is not traversed

        if (curr == numberOfNodes - 1) {
            return fitnessValue[curr];
        }

        if (outDegree[curr] == 1) {
            return fitnessValue[curr] + traverseCurrentPath(adj[curr][0], visitedEdge, path);
        }

        if (visitedEdge[curr].find(path[curr]) == visitedEdge[curr].end()) {
            visitedEdge[curr].insert(path[curr]);
            return fitnessValue[curr] + traverseCurrentPath(adj[curr][path[curr]], visitedEdge, path);
        }

        if (visitedEdge[curr].size() == outDegree[curr]) {
            int maxmFitnessFromRestNodes = 0;
            for (auto i : adj[curr]) {
                if (i != curr) {
                    int decision = path[i];
                    for (int j = outDegree[i] - 1; j >= 0; j--) {
                        if (j != decision) {
                            path[i] = j;
                            maxmFitnessFromRestNodes = max(maxmFitnessFromRestNodes, traverseCurrentPath(i, visitedEdge, path));
                        }
                    }
                    path[i] = decision;
                }
            }
            return maxmFitnessFromRestNodes;
        }

        int maxReverseHeight = 0;
        int maxReverseHeightIndex = 0;
        for (int i = 0; i < outDegree[curr]; i++) {
            if (visitedEdge[curr].find(i) == visitedEdge[curr].end()) {
                if (reverseHeight[adj[curr][i]] > maxReverseHeight) {
                    maxReverseHeight = reverseHeight[adj[curr][i]];
                    maxReverseHeightIndex = i;
                }
            }
        }
        visitedEdge[curr].insert(maxReverseHeightIndex);
        return fitnessValue[curr] + traverseCurrentPath(adj[curr][maxReverseHeightIndex], visitedEdge, path);
    }

    // Function to traverse the graph and calculate the fitess value of a given path (test case) based on the combination of decision nodes
    void traverseAllPaths(){
        generateTestPaths(0, {});
        for (auto& path : paths) {
            vector<unordered_set<int>> visitedEdge(numberOfNodes);
            pathFitness[path] = max(0, traverseCurrentPath(0, visitedEdge, path));
        }
    }

    int calculateDecisionNodes(){
        int decisionNodes = 0;
        for (int i = 0; i < numberOfNodes; i++) {
            if (outDegree[i] > 1) {
                decisionNodes++;
            }
        }
        return decisionNodes;
    }

    void printPathFitness(){
        int decisionNodes = calculateDecisionNodes();
        cout << paths.size() << " " << decisionNodes << endl;
        for (auto path : paths) {
            for (int i = 0; i < path.size(); i++) {
                if (outDegree[i] > 1) {
                    cout << path[i] + 1 << " ";
                }
            }
            cout << pathFitness[path] << endl;
        }
    }

    void printNodeFitness(){
        for (int i = 0; i < numberOfNodes; i++) {
            cerr << i + 1 << ": " << stackBasedWeight[i] << " " << informationGain[i] << " " << fitnessValue[i] << endl;
        }
    }

    //Getters

    map<vector<int>, int> getPathFitness(){
        return pathFitness;
    }

    vector<int> getOutDegree(){
        return outDegree;
    }
};

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

    // void unitTesting(){
    //     vector<string> a;
    //     for(int i = 0; i < pow(2, 8); i++){
    //         string s = "";
    //         for(int j = 0; j < 8; j++){
    //             if(i & (1 << j))
    //                 s += '1';
    //             else
    //                 s += '0';
    //         }
    //         a.push_back(s);
    //     }
    //     for(auto i: a){
    //         if(isChromosomeInBounds(i)){
    //             if(pathFitness.find(i) != pathFitness.end()){
    //                 cout << i << endl;
    //             }
    //         }
    //     }
    // }

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

int main()
{

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.csv", "w", stdout);
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
    
    for(int hyperparameters = 2; hyperparameters <= n/2; hyperparameters++){
        for(int maxIterations = 2; maxIterations <= n; maxIterations++){
            cout<<"N,"<<hyperparameters<<","<<"IT,"<<maxIterations<<",";
            int trials = 1000;
            double averageGrasshopperFitness = 0;
            double averageGeneticFitness = 0;
            while(trials--){
                GrassHopper grasshopper(n, edges, hyperparameters, maxIterations, 1.19999999999999995559, 0.80000000000000004441, 1.00000000000000000000, 0.00010000000000000000);  
                averageGrasshopperFitness = grasshopper.getBestFitness();  
                Genetic genetic(n, edges, hyperparameters, maxIterations);
                averageGeneticFitness = genetic.getBestFitness();
            }
            cout<<averageGrasshopperFitness<<","<<averageGeneticFitness<<endl;
        }
    }

    int hyperparameters = 6;
    int maxIterations = 12;
    for(double f = 0; f <= 2; f += 0.1){
        for(double l = 0; l <= 2; l += 0.1){
            for(double cmin = 0; cmin <= 2; cmin += 0.1){
                cout<<"F,"<<f<<","<<"L,"<<l<<","<<"C,"<<cmin<<",";
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
    
    return 0;
}