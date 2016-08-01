#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <string>

#include <math.h>
#include <stdio.h>

using namespace std;

#define SIGMOID_BOUND 6

const int sigmoid_table_size = 1000;

typedef float real;

real* sigmoid_table;
real sample_ratio = 0.01;
int total_dim, cutoff = 0;

map<string, vector<real> > wordAEmb;
map<string, vector<real> > wordBEmb;

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

real FastSigmoid(real x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

bool my_getline(ifstream &inf, string &line) {
    if (!getline(inf, line))
        return false;
    int end = line.size() - 1;
    while (end >= 0 && (line[end] == '\r' || line[end] == '\n')) {
        line.erase(end--);
    }
    return true;
}

void split_bychars(const string& str, vector<string> & vec, const char *sep = " ") {
    vec.clear();
    string::size_type pos1 = 0, pos2 = 0;
    string word;
    while ((pos2 = str.find_first_of(sep, pos1)) != string::npos) {
        word = str.substr(pos1, pos2 - pos1);
        pos1 = pos2 + 1;
        if (!word.empty())
            vec.push_back(word);
    }
    word = str.substr(pos1);
    if (!word.empty())
        vec.push_back(word);
}

void GetNodeLis(char *fileName, vector<string> &nodeLis){
    ifstream fin;
    fin.open(fileName);

    int cnt = 0;
    string line;
    while(my_getline(fin, line)){
        nodeLis.push_back(line);
        
        if(cnt++ % 10000 == 0){
            printf("Had processed %d nodes%c", cnt, 13);
            fflush(stdout);
        }
    }
    fin.close();
}

void GetWordMultiEmbs(char *embFile, map<string, vector<real> >& wordEmbs){
    ifstream fin;
    fin.open(embFile);

    string line;
    int cnt = 0;
    string preKey = "";
    string key = "";
    vector<real> emb;

    while(my_getline(fin, line)){
        if(cnt++ % 10000 == 0){
            printf("Had processed %d lines %c", cnt, 13);
            fflush(stdout);
        }

        vector<string> strLis;
        split_bychars(line, strLis);
        
        if(strLis.size() < 10)
            continue;
        else{
            
            key = strLis[0];
            if(preKey != "" && preKey != key){
                if(wordEmbs.count(preKey) == 0){
                    vector<real> embCopy(emb);
                    wordEmbs.insert(std::pair<string, vector<real> >(preKey, embCopy));
                }
                preKey = key;
                emb.clear();
            }

            
            for(int i = 0; i < strLis.size() - 1; i++){
                emb.push_back(atof(strLis[i + 1].c_str()));
            }
            preKey = key;
        }
    }

    if(wordEmbs.count(key) == 0)
        wordEmbs.insert(std::pair<string, vector<real> >(key, emb));
}

void GetWordEmbs(char *embFile, map<string, vector<real> >& wordEmbs){
    ifstream fin;
    fin.open(embFile);
    
    string line;
    int cnt = 0;
    while(my_getline(fin, line)){
        if(cnt++ % 10000 == 0){
            printf("Had processed %d lines%c", cnt, 13);
            fflush(stdout);
        }

        vector<string> strLis;
        split_bychars(line, strLis);

        if(strLis.size() < 10)
            continue;
        else{
            string key = strLis[0];
            vector<real> emb;
            for(int i = 0; i < strLis.size() - 1; i++){
                emb.push_back(atof(strLis[i + 1].c_str()));
            }

            if(wordEmbs.count(key) == 0){
                wordEmbs.insert(std::pair<string, vector<real> >(key, emb));
            }
        }
    }

    fin.close();
}

void GetSet(char *trainFile, char *testFile, vector<string>& trainSet, vector<string>& testSet){
    ifstream finTrain, finTest;
    
    finTrain.open(trainFile); 
    finTest.open(testFile);

    string line;
    while(my_getline(finTrain, line)){
        vector<string> strLis;
        
        split_bychars(line, strLis);

        string key = strLis[0] + " " + strLis[1];
        trainSet.push_back(key);
    }
    finTrain.close();

    while(my_getline(finTest, line)){
        vector<string> strLis;
        split_bychars(line, strLis);

        string key = strLis[0] + " " + strLis[1];
        testSet.push_back(key);
    }
    finTest.close();

}

/*int AssignByNum(string u, int dim){
    int k;
    int max_num = -1000;
    int num_sense = wordBEmb[u].size() / (2 + dim);

    for(int i = 0; i < num_sense; i++){
        if (max_num < wordBEmb[u][i * (dim + 2) + 1]){
            max_num = wordBEmb[u][i * (dim + 2) + 1];
            k = i;
        }
    }
    //cout << "k" << k << endl;
    //cin.get();
    return k;
}*/

/*int AssignByContext(string u, string v){
    int dim = wordAEmb[v].size();
    int num_sense = wordBEmb[u].size() / (2 + dim);
    //cout << num_sense << endl; cin.get();

    double max_sim = -100000.0;
    int k;

    for(int i = 0; i < num_sense; i++){
    	for (int j=0; j < num_sense; j++){
		double sum, sum1, sum2;
        	sum = sum1 = sum2 = 0;

        	for(int d = 0; d < dim; d++){
        		sum += wordBEmb[u][k * (2 + dim) + 2 + d] * wordAEmb[v][d];
        		sum1 += pow(wordBEmb[u][k * (2 + dim) + 2 + d], 2.0);
        		sum2 += pow(wordAEmb[v][d], 2.0);
        	}

        double sim = sum / (sqrt(sum1) * sqrt(sum2) + 1e-8);

        if(sim > max_sim){
            k = i;
            max_sim = sim;
        }
    }
    return k;

}

int Assign(string u, string v){
    return AssignByContext(u, v);

    int dim = wordCEmb[v].size();
    return AssignByNum(u, dim);

}

real LinkPredictionByOneSense(string u, string v){
	real x;
	int dim = wordAEmb[v].size();

	int k_sense = Assign(u, v);
	real sum, sum1, sum2;
	sum = sum1 = sum2 = 0;
	for (int k = 0; k < dim; k++){
		sum += wordAEmb[u][k_sense * (dim + 2) + 2 + k] * wordAEmb[v][k];
		sum1 += pow(wordAEmb[u][k_sense * (dim + 2) + 2 + k], 2.0);
		sum2 += pow(wordAEmb[v][k], 2.0);
	}

	x = sum / (sqrt(sum1) * sqrt(sum2) + 1e-8);
	return x;
}*/

real LinkPredictionByMultiSense(string u, string v){
	real x = 0;
	int dim = total_dim;
	
	int num_sense = wordAEmb[u].size() / (2 + dim);
	int num_sense_2 = wordAEmb[v].size() / (2 + dim);
	int num_token = 0;
    int num_sense_token = 0;

	real min, max;
	min = 10000;
	max = -10000;

	for (int i = 0; i < num_sense ; i++){
		num_sense_token = wordAEmb[u][i * (dim + 2) + 1];
        if(num_sense_token < cutoff)
            continue;

        for (int j = 0; j < num_sense_2; j++){
		    num_sense_token = wordAEmb[v][j * (dim + 2) + 1];
            if(num_sense_token < cutoff)
                continue;

			num_sense_token = 1;
			num_token += num_sense_token;

			real sum, sum1, sum2;
			sum = sum1 = sum2 = 0;

			for (int k = 0; k < dim; k++){
				sum += wordAEmb[u][i * (dim + 2) + 2 + k] * wordAEmb[v][j * (dim + 2) + 2 + k];
				sum1 += pow(wordAEmb[u][i * (dim + 2) + 2 + k], 2.0);
				sum2 += pow(wordAEmb[v][j * (dim + 2) + 2 + k], 2.0);
			}

			real tmp = sum * num_sense_token / (sqrt(sum1) * sqrt(sum2));
			x += tmp;

			if (tmp < min){
				min = tmp;
			}

			if (tmp > max){
				max = tmp;
			}
		}
	}
    
    if(num_token == 0)
        return 0;
	x /= num_token;
	//return max;
	return x;
}

real LinkPrediction(string u, string v){
	return LinkPredictionByMultiSense(u, v);
}

int main(int argc, char** argv){
    InitSigmoidTable();
    cout << "InitSigmoid done!\n";
    
    vector<string> nodeLis;
    GetNodeLis(argv[1], nodeLis);
    cout << "\nGet nodes done!\n";

    GetWordMultiEmbs(argv[2], wordAEmb);
    cout << "\nGet A embs done!\n";
    
    GetWordMultiEmbs(argv[3], wordBEmb);
    cout << "\nGet B embs done!\n";
    
    vector<string> trainSet; vector<string> testSet;
    GetSet(argv[4], argv[5], trainSet, testSet);
    cout << "\nGet set done!\n";
    
    ofstream foutTestSet, foutMisSet;
    foutTestSet.open(argv[6]); foutMisSet.open(argv[7]);

    total_dim = atoi(argv[8]);
    cutoff = atoi(argv[9]);    

    int ccnt = 0;
    int n_p = 0;
    int n_pp = 0;
    real misV = 0.0;
    real epV = 0.0;
    real auc = 0.0;

    for(int i = 0; i < testSet.size(); i++){
        string key = testSet[i];
        string u,v;

        vector<string> items;
        split_bychars(key, items);
        u = items[0]; v = items[1];
        
        real x = 0;
        
        if(wordAEmb.count(u) == 0){
            epV = 0.0;    
        }
        else
        {
			epV = LinkPrediction(u, v);
        }

        string us,vs;
        int nodeSize = nodeLis.size();
        us = nodeLis[rand()%nodeSize];
        vs = nodeLis[rand()%nodeSize];

        key = us + " " + vs;
        
        x = 0;
        if(wordAEmb.count(us) == 0){
            misV = 0.0;
        }
        else
        {
            misV = LinkPrediction(us, vs);
        }
        
        ccnt++;
        if(epV > misV)
            n_p++;
        else if(epV == misV)
            n_pp++;

        auc = (0.5*n_pp + 1.0*n_p) / ccnt;
        printf("Had processed : ccnt %d, n_p %d, n_pp %d, auc %lf%%%c", ccnt, n_p, n_pp, auc * 100, 13);
        fflush(stdout);
    
    }

    cout << "\nDone " << endl;
    return 0;
    while(true){}
    for(int i = 0; i < nodeLis.size(); i++){

        if(i % 1 == 0){
            auc = (0.5*n_pp + 1.0*n_p) / ccnt;
            printf("Had processed %d nodes; ccnt %d, n_p %d, n_pp %d, auc %lf%%%c", i, ccnt, n_p, n_pp, auc, 13);
            fflush(stdout);
        }

        for(int j = 0; j < nodeLis.size(); j++){
            string key = nodeLis[i] + " " + nodeLis[j]; 
            if(wordAEmb.count(nodeLis[i]) == 0 || wordBEmb.count(nodeLis[j]) == 0) 
            {
                foutMisSet << nodeLis[i] << " " << nodeLis[j] << " 0.5";
                misV = 0.5;
            }
            else if(find(trainSet.begin(), trainSet.end(),key) != trainSet.end()){
                continue;
            }
            else if(find(testSet.begin(), testSet.end(), key) != testSet.end()){
                real x = 0;
             
                for(int k = 0; k < wordAEmb[nodeLis[i]].size(); k++){
                    x += wordAEmb[nodeLis[i]][k] * wordBEmb[nodeLis[j]][k];
                }
                epV = FastSigmoid(x);
                
                ccnt++;
                if(epV > misV)
                    n_p++;
                else if(epV == misV)
                    n_pp++;

                foutTestSet << key << " " << FastSigmoid(x) << endl;
            }
            else{

                //if((rand() / (real)RAND_MAX) < 1 - sample_ratio)
                //    continue;

                real x = 0;
                for(int k = 0; k < wordAEmb[nodeLis[i]].size(); k++){
                    x += wordAEmb[nodeLis[i]][k] * wordBEmb[nodeLis[j]][k];
                }

                misV = FastSigmoid(x);
                foutMisSet << key << " " << FastSigmoid(x) << endl;
            }
        }
    }
    foutTestSet.close();
    foutMisSet.close();

    cout << "Writing done!\n";
    return 0;
}
