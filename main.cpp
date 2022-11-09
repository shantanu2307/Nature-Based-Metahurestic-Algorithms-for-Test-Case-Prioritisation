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
            bool flag=false;
            for(auto j:adj[i]){
                if(j==i){
                    flag=true;
                    break;
                }
            }
            if(flag){
                informationGain.push_back((inDegree[i]-1)*(outDegree[i]-1));
            }
            else{
                informationGain.push_back(inDegree[i] * outDegree[i]);
            }
        }
    }

    void bfs(vector<set<vector<int>>>&res){
        vector<bool>vis(numberOfNodes, 0);
        queue<int> q;
        q.push(0);
        int level=0;
        vis[0] = true;
        while(!q.empty()){
            int sz=q.size();
            set<vector<int>>nkCombination;
            while(sz--){
                int curr=q.front();
                q.pop();
                for(int node:adj[curr]){
                    if(!vis[node]){
                        vis[node]=true;
                        q.push(node);
                        nkCombination.insert({node, level+1});
                    }
                    if(node==(numberOfNodes-1)){
                        nkCombination.insert({node, level+1});
                    }
                }
            }
            if(!nkCombination.empty()){
                res.push_back(nkCombination);
            }
            level++;
        }
    }

    // Function to calculate the stack based weight of a graph
    void calculateStackBasedWeight(){
        // weight of current node = maximum height of any node - minimum depth of current node
        int maxDepth = 0;
        for (int i = 0; i < numberOfNodes; i++) {
            maxDepth = max(maxDepth, reverseHeight[i]);
        }
        maxDepth++;
        vector<set<vector<int>>>res;
        bfs(res);
        stackBasedWeight[0]=maxDepth;
        for(auto i:res){
            for(auto j:i){
                stackBasedWeight[j[0]]+=maxDepth-j[1];
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
        cout<<"Node,"<<"Stack Based Weight,"<<"Information Gain,"<<"Fitness Value"<<endl;
        for (int i = 0; i < numberOfNodes; i++) {
            cout << i + 1 << "," << stackBasedWeight[i] << "," << informationGain[i] << "," << fitnessValue[i] << endl;
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

class Cheetah{

  int numberOfNodes;
  int dimension;
  int numberOfCheetahs;
  int numberOfSearchAgents;
  int maxIterations;
  map<vector<int>, int> pathFitness;
  vector<vector<int>>paths;
  vector<int>upperBound;
  vector<int>lowerBound;
  Graph *graph;  
  public:
  Cheetah(int numberOfNodes, vector<pair<int, int>>&edges, int numberOfCheetahs, int numberOfSearchAgents, int maxIterations){
    graph=new Graph(numberOfNodes, edges);
    vector<int> outDegree = graph -> getOutDegree();
    map<vector<int>, int> pFitness = graph -> getPathFitness();
    this->numberOfNodes = numberOfNodes;
    this->numberOfCheetahs=numberOfCheetahs;
    this->numberOfSearchAgents=numberOfSearchAgents;
    this->maxIterations=maxIterations;
    // Calculating Dimension
    int dimesnion = 0;
    for (int i = 0; i < numberOfNodes; i++) {
        if (outDegree[i] > 1) {
            upperBound.push_back(outDegree[i]+1);
            lowerBound.push_back(1);
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
    initialise();
  }

  void initialise(){
    set<vector<int>>initialPopulation;
    int bestSolution=0;
    vector<int>bestSolutionPosition;

    while(initialPopulation.size()<numberOfCheetahs){
        //select a random path from paths vector
        int randomPathIndex = rand()%paths.size();
        vector<int>randomPath = paths[randomPathIndex];
        initialPopulation.insert(randomPath);
    }

    vector<vector<int>>pop(initialPopulation.begin(), initialPopulation.end());
    vector<vector<int>>home=pop;

    
    // Finding the leader posintion
    for(auto i: pop){
        if(pathFitness[i]>bestSolution){
            bestSolution=pathFitness[i];
            bestSolutionPosition=i;
        }
    }


    vector<int>X_best=bestSolutionPosition;

    int t=0;
    int it=1;
    int maxIt=this->maxIterations;
    vector<int>Golbest(maxIt, 0);
    vector<int>bestCost(maxIt, 0);
    int T=ceil(double(this->dimension)/10.00)*60;
    while(it<=maxIt){
        // choose m random cheetas from initial population
        set<vector<int>>mRandomCheetahs;

        while(mRandomCheetahs.size()<numberOfSearchAgents){
            int randomCheetahIndex = rand()%initialPopulation.size();
            auto it = initialPopulation.begin();
            advance(it, randomCheetahIndex);
            mRandomCheetahs.insert(*it);
        }

    
        vector<vector<int>>mRandomCheetahsVector(mRandomCheetahs.begin(), mRandomCheetahs.end());
        
        
        for(int l=0;l<mRandomCheetahsVector.size();l++){
            // choose a random neighbour of i in mRandomCheetahs
            int randomNeighbourIndex = rand()%mRandomCheetahs.size();
            vector<int> randomNeighbour = mRandomCheetahsVector[randomNeighbourIndex]; // Neighbour Posn
            vector<int> Xb=bestSolutionPosition; // Leader Posn
            vector<int> Xbest=X_best; // Prey Posn
            vector<int> X=mRandomCheetahsVector[l]; // Cheetah Posn

            double kk=0;
            if(randomNeighbourIndex<=1 && t>2 && t>ceil(0.2*T+1) && (abs(bestCost[t-2] - bestCost[t-ceil(0.2*T+1)]))<=0.0001*Golbest[t-1]){
                X=X_best;
                kk=0;
            }
            else if(randomNeighbourIndex==2){
                X=bestSolutionPosition;
                kk = -0.1 * (rand()%100)/100.00 * t/T;
            }
            else{
                kk=0.25;
            }

            vector<int> Z=X;
            vector<int>randomTestCase;
            for(int i=0;i<dimension;i++){
                // generate a random number between lowerBound and upperBound
                int randomNum = abs(rand()%(upperBound[i]-1))+1;
                randomTestCase.push_back(randomNum);
            }
            assert(pathFitness.find(randomTestCase) != pathFitness.end());

            for(int d=0;d<dimension;d++){
                double rHat=rand()%10;
                double r1=(rand()%100)/100.00;
                double alpha;
                if(l==0){ // leader
                    alpha=0.0001*t/T*(upperBound[d]-lowerBound[d]);
                }
                else{  // member
                    alpha=0.0001*t/T*abs(Xb[d]-X[d]+1) + 0.001*round(double((rand()%100/100)>0.9));
                }
                double r=rand()%3;
                double p=exp(r/2)*sin(2*3.14*r);
                double rCheck=pow(abs(r), p); // turning factor
                double beta=randomNeighbour[d]-X[d]; // interaction factor
                double h0=exp(2.00 - 2.00*double(t)/double(T));
                double H=abs(2*r1*h0 -h0);
                double r2=(rand()%100)/100.00;
                double r3=kk+(rand()%100)/100.00;
                // Strategy selection
                if(r2<r3){
                    double r4=3.00 *(rand()%100)/100.00;
                    if(H>r4){
                        Z[d]=(X[d]+round(alpha/rHat)); // search
                    }
                    else{
                        Z[d]=Xbest[d]+round(rCheck*beta); // attack
                    }  
                }
                else{
                    Z[d]=X[d]; // sit and wait
                }

            }
            for(int i=0;i<dimension;i++){
                if(Z[i]<lowerBound[i]){
                    Z[i]=rand()%(upperBound[i]-lowerBound[i]+1)+lowerBound[i];
                }
                else if(Z[i]>upperBound[i]){
                    Z[i]=rand()%(upperBound[i]-lowerBound[i]+1)+lowerBound[i];
                }
            }
            vector<int>newSolution=Z;
            int newSolutionFitness=pathFitness[newSolution];
            if(newSolutionFitness>pathFitness[randomNeighbour]){
                pop[randomNeighbourIndex]=newSolution;
                if(newSolutionFitness>bestSolution){
                    bestSolution=newSolutionFitness;
                    bestSolutionPosition=newSolution;
                }

            }
        }

        t++;

        if(t>T && t>2 && (t-round(T)-1>=0)){
            if(abs(bestCost[t-1] - bestCost[t-round(T)-1])<=abs(0.01*bestCost[t-1])){
                vector<int>best=X_best;
                int j0=rand()%dimension;
                best[j0]=rand()%(upperBound[j0]-1)+1;
                bestSolution=pathFitness[best];
                bestSolutionPosition=best;
                // generate m differenct random integers between 0, n-1
                set<int>randomIntegers;
                while(randomIntegers.size()<numberOfSearchAgents){
                    int randomInteger = rand()%numberOfCheetahs;
                    randomIntegers.insert(randomInteger);
                }
                int ctr=0;
                for(int i=numberOfCheetahs-numberOfSearchAgents;i<numberOfCheetahs;i++){
                    // select ith random integer from randomIntegers
                    auto it = randomIntegers.begin();
                    advance(it, i);
                    int randomInteger = *it;
                    it=randomIntegers.begin();
                    advance(it, ctr);
                    int randomInteger2 = *it;
                    pop[randomInteger]=home[randomInteger2];
                    ctr++;
                }
                t=1;
            }
        }

        it++;
        if(bestSolution>pathFitness[X_best]){
            X_best=bestSolutionPosition;
        }
        bestCost[t]=bestSolution;
        for(int i=0;i<=t;i++){
            Golbest[i]=pathFitness[X_best];
        }
    }
    cout<<"Best Solution: Cheetah "<<pathFitness[X_best]<<endl;
  }

};

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
    g.printNodeFitness();
    g.printPathFitness();
}

int main()
{

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

#ifndef ONLINE_JUDGE
    freopen("input2.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    srand(time(0)); 
    // compareAverageFitness();
    // printFitnessValues();
    // getCountsWhereGrassHopperDominatesGenetic();
    int n, e;
    cin >> n >> e;
    vector<pair<int, int>> edges;
    for (int i = 0; i < e; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({ u - 1, v - 1 });
    }
    GrassHopper goa(n, edges, 6, 10 ,1.19999999999999995559, 0.80000000000000004441, 1.00000000000000000000, 0.00010000000000000000);
    Cheetah c(n, edges, 6, 3, 10);
    Genetic g(n, edges, 6, 10);
    cout<<"Best Solution: GrassHopper "<<goa.getBestFitness()<<endl;
    cout<<"Best Solution: Genetic "<<g.getBestFitness()<<endl;
    return 0;
}