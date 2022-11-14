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

#pragma once

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
        // traverseAllPaths();
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
        cout<<maxDepth<<endl;
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