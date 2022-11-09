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

class Cheetah{

  int numberOfNodes;
  int dimension;
  int numberOfCheetahs;
  int numberOfSearchAgents;
  int maxIterations;
  int bestFitness;
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
    int dimension = 0;
    for (int i = 0; i < numberOfNodes; i++) {
        if (outDegree[i] > 1) {
            upperBound.push_back(outDegree[i]);
            lowerBound.push_back(1);
            dimension++;
        }
    }
    this->dimension = dimension;
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
    assert(pop.size()==numberOfCheetahs);
    vector<vector<int>>home=pop;
    assert(home.size()==numberOfCheetahs);
    
    // Finding the leader posintion
    for(auto i: pop){
        if(pathFitness[i]>bestSolution){
            bestSolution=pathFitness[i];
            bestSolutionPosition=i;
        }
    }

    vector<int>X_best=bestSolutionPosition;
    assert(X_best.size()==dimension);

    int t=0;
    int it=0;
    int maxIt=this->maxIterations;
    vector<int>Golbest(100005, 0);
    vector<int>bestCost(100005, 0);
    int T=ceil(double(this->dimension)/10.00)*60;
    assert(T>0);

    while(it<maxIt){
        // choose m random cheetas from initial population
        set<vector<int>>mRandomCheetahs;

        while(mRandomCheetahs.size()<numberOfSearchAgents){
            int randomCheetahIndex = rand()%initialPopulation.size();
            auto it = initialPopulation.begin();
            advance(it, randomCheetahIndex);
            assert(it!=initialPopulation.end());
            mRandomCheetahs.insert(*it);
        }

    
        vector<vector<int>>mRandomCheetahsVector(mRandomCheetahs.begin(), mRandomCheetahs.end());
        assert(mRandomCheetahsVector.size()==numberOfSearchAgents);
        
        for(int l=0;l<mRandomCheetahsVector.size();l++){
            // choose a random neighbour of i in mRandomCheetahs
            int randomNeighbourIndex = rand()%mRandomCheetahs.size();
            assert(randomNeighbourIndex>=0 && randomNeighbourIndex<mRandomCheetahs.size());

            vector<int> randomNeighbour = mRandomCheetahsVector[randomNeighbourIndex];// Neighbour Posn
            assert(randomNeighbour.size()==dimension);
            vector<int> Xb=bestSolutionPosition; // Leader Posn
            assert(Xb.size()==dimension);
            vector<int> Xbest=X_best; // Prey Posn
            assert(Xbest.size()==dimension);
            vector<int> X=mRandomCheetahsVector[l]; // Cheetah Posn
            assert(X.size()==dimension);
            
            double kk=0;
            if(randomNeighbourIndex<=1 && t>=2 && t>ceil(0.2*T+1) && (abs(bestCost[t-2] - bestCost[t-ceil(0.2*T+1)]))<=0.0001*Golbest[t-1]){
                assert(t-2>=0 && t-ceil(0.2*T+1)>=0 && t-ceil(0.2*T+1)<bestCost.size() && t<maxIt);
                X=X_best;
                kk=0;
            }
            else if(randomNeighbourIndex==2){
                assert(T>0);
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
                int randomNum = abs(rand()%(upperBound[i]))+1;
                randomTestCase.push_back(randomNum);
            }
            
            assert(pathFitness.find(randomTestCase) != pathFitness.end());

            for(int d=0;d<dimension;d++){
                double rHat=rand()%10+1;
                double r1=(rand()%100)/100.00;
                double alpha;
                if(l==0){ // leader
                    alpha=0.0001*t/T*(upperBound[d]-lowerBound[d]);
                }
                else{  // member
                    alpha=0.0001*(t/T)*abs(Xb[d]-X[d]+1) + 0.001*round(double((rand()%100/100)>0.9));
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
                        assert(rHat!=0);
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
                    Z[i]=rand()%(upperBound[i])+1;
                }
                else if(Z[i]>upperBound[i]){
                    Z[i]=rand()%(upperBound[i])+1;
                }
                assert(Z[i]<=upperBound[i] && Z[i]>=lowerBound[i]);
            }

            vector<int>newSolution=Z;
            assert(newSolution.size()==dimension);
            assert(pathFitness.find(newSolution) != pathFitness.end());

            int newSolutionFitness=pathFitness[newSolution];
            assert(newSolutionFitness>=0);
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
                assert(t-round(T)-1>=0 && t-round(T)-1<bestCost.size() && t-1>=0 && t-1<bestCost.size());
                vector<int>best=X_best;
                int j0=rand()%dimension;
                assert(j0>=0 && j0<dimension);
                best[j0]=rand()%(upperBound[j0])+1;
                assert(pathFitness.find(best) != pathFitness.end());
                bestSolution=pathFitness[best];
                bestSolutionPosition=best;
                // generate m differenct random integers between 0, n-1
                set<int>randomIntegers;
                while(randomIntegers.size()<numberOfSearchAgents){
                    int randomInteger = rand()%numberOfCheetahs;
                    randomIntegers.insert(randomInteger);
                    assert(randomInteger>=0 && randomInteger<numberOfCheetahs);
                }
                assert(randomIntegers.size()==numberOfSearchAgents);
                int ctr=0;
                for(int i=numberOfCheetahs-numberOfSearchAgents;i<numberOfCheetahs;i++){
                    // select ith random integer from randomIntegers
                    auto it = randomIntegers.begin();
                    advance(it, i);
                    assert(it != randomIntegers.end());
                    int randomInteger = *it;
                    assert(randomInteger>=0 && randomInteger<numberOfCheetahs);
                    it=randomIntegers.begin();
                    advance(it, ctr);
                    assert(it != randomIntegers.end());
                    int randomInteger2 = *it;
                    assert(randomInteger2>=0 && randomInteger2<numberOfCheetahs);
                    pop[randomInteger]=home[randomInteger2];
                    ctr++;
                }
                t=0;
            }
        }

        it++;
        if(bestSolution>pathFitness[X_best]){
            X_best=bestSolutionPosition;
        }
        
        assert(t>=0 && t<bestCost.size());
        bestCost[t]=bestSolution;
       
        for(int i=0;i<t;i++){
            Golbest[i]=pathFitness[X_best];
        }
    }
    this->bestFitness=pathFitness[X_best];
  }

  int getBestFitness(){
    return this->bestFitness;
  }

};