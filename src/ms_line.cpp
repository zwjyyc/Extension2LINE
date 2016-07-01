/*
This is the tool ....

*/

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>

#include <iostream>

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

using namespace std;

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {
	double degree;
	char *name;
};

char network_file[MAX_STRING], embedding_file[MAX_STRING];
struct ClassVertex *vertex;
int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;
int num_sense = -1, max_num_sense = 10;
int *vertex_hash_table, *neg_table;
int max_num_vertices = 1000, num_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges = 0;
real init_rho = 0.025, rho, gap = -0.5;
real *emb_context, *sigmoid_table;
real **multi_sense_emb, **multi_cluster_emb;

int *edge_source_id, *edge_target_id;
double *edge_weight;

// Parameters for edge sampling
long long *alias;
double *prob;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key)
	{
		hash = hash * seed + (*key++);
	}
	return hash % hash_table_size;
}

void InitHashTable()
{
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value)
{
	int addr = Hash(key);
	while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
	vertex_hash_table[addr] = value;
}

int SearchHashTable(char *key)
{
	int addr = Hash(key);
	while (1)
	{
		if (vertex_hash_table[addr] == -1) return -1;
		if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
		addr = (addr + 1) % hash_table_size;
	}
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	strcpy(vertex[num_vertices].name, name);
	vertex[num_vertices].degree = 0;
	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices - 1);
	return num_vertices - 1;
}

/* Read network from the training file */
void ReadData()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0;
	while (fgets(str, sizeof(str), fin)) num_edges++;
	fclose(fin);
	printf("Number of edges: %lld          \n", num_edges);

	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	fin = fopen(network_file, "rb");
	num_vertices = 0;
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);
		if (vid == -1) vid = AddVertex(name_v1);
		vertex[vid].degree += weight;
		edge_source_id[k] = vid;

		vid = SearchHashTable(name_v2);
		if (vid == -1) vid = AddVertex(name_v2);
		vertex[vid].degree += weight;
		edge_target_id[k] = vid;

		edge_weight[k] = weight;
	}
	fclose(fin);
	printf("Number of vertices: %d          \n", num_vertices);
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable()
{
	alias = (long long *)malloc(num_edges*sizeof(long long));
	prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b, k;

	//a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real));
	multi_sense_emb = (real **)malloc(num_vertices * sizeof(real *));
	multi_cluster_emb = (real **)malloc(num_vertices * sizeof(real *));

	if (multi_sense_emb == NULL || multi_cluster_emb == NULL)
    { printf("Error: memory allocation failed\n"); exit(1); }
    
    for (a = 0 ; a < num_vertices; a++){
        if (num_sense == -1){

			multi_sense_emb[a] = (real *)malloc(dim * max_num_sense * sizeof(real));
			multi_cluster_emb[a] = (real *)malloc((1 + (dim + 1) * max_num_sense) * sizeof(real));
			multi_cluster_emb[a][0] = 1;
			multi_cluster_emb[a][1] = 0;
			for (b = 0; b < dim; b++){
				multi_sense_emb[a][b] = (rand() / (real)RAND_MAX - 0.5) / dim;
			}
            
            for (k = 0; k < max_num_sense; k++){
                for(b = 0; b < dim; b++){
                    multi_cluster_emb[a][2 + (dim + 1)*k + b] = 0;
                }

                multi_cluster_emb[a][1 + (dim + 1)*k] = 0;
            }
        }
        else
        {
            multi_sense_emb[a] = (real *)malloc(dim * num_sense * sizeof(real));
			multi_cluster_emb[a] = (real *)malloc((1 + (dim + 1) * num_sense) * sizeof(real));
			multi_cluster_emb[a][0] = num_sense;

            for (k = 0; k < num_sense; k++){
                for (b = 0; b < dim; b++){
                    multi_sense_emb[a][k * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
                }
                int la = 1 + k * (dim + 1);
				multi_cluster_emb[a][la] = 0;
            }
        }
    }

	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

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

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

// only one node for context June 25, 2016
void AssignContextVec(int v, real *v_cxt)
{
    long long lv = v * dim;
    for(int c = 0; c != dim; c++) v_cxt[c] = emb_context[c + lv];
}

// useful ?
int AssignSenseBySample(real *sims)
{
    return 0;
}

void CreatNewClusterV2(int u)
{
	int num_word_sense = (int)multi_cluster_emb[u][0];
	int lu = 1 + num_word_sense * (dim + 1), i;

	num_word_sense++;
	multi_cluster_emb[u][lu] = 0;
	
    for (i = (num_word_sense - 1) * dim; i < num_word_sense * dim; i++){
		multi_sense_emb[u][i] = (rand() / (real)RAND_MAX - 0.5) / dim;
	}
	
    multi_cluster_emb[u][0] += 1;
}

void CreatNewCluster(int u)
{
    int num_word_sense = (int)multi_cluster_emb[u][0];
    int lu = 1 + num_word_sense * (dim + 1), i;
	real *tmpPtr1 = multi_cluster_emb[u];
	real *tmpPtr2 = multi_sense_emb[u];

	num_word_sense++;

	printf("num_sense : %d\n", num_word_sense);
	fflush(stdout);
	//multi_cluster_emb[u] = (real *)malloc((num_word_sense * (dim + 1) + 1) * sizeof(real));
	//multi_sense_emb[u] = (real *)malloc(num_word_sense * dim * sizeof(real));

    multi_cluster_emb[u] = (real *)realloc(multi_cluster_emb[u], (num_word_sense * (dim + 1) + 1) * sizeof(real));
    multi_sense_emb[u] = (real *)realloc(multi_sense_emb[u], num_word_sense * dim * sizeof(real));
	printf("ok!\n"); fflush(stdout);
    if(multi_cluster_emb[u] != NULL && multi_sense_emb[u] != NULL){
		//memcpy(multi_cluster_emb[u], tmpPtr1, ((num_word_sense - 1) * (dim + 1) + 1) * sizeof(real));
		//memcpy(multi_sense_emb[u], tmpPtr2, (num_word_sense - 1) * dim * sizeof(real));

        multi_cluster_emb[u][lu] = 0;
        for(i = (num_word_sense - 1) * dim; i < num_word_sense * dim; i++){
            multi_sense_emb[u][i] = (rand() / (real)RAND_MAX - 0.5) / dim;
        }
		printf("ok1!\n"); fflush(stdout);
    }
    else {
        puts("Error (re)allocating memory");
        exit(1);
    }

	printf("ok2!\n"); fflush(stdout);
	free(tmpPtr1); free(tmpPtr2);
}

int FindNearestCluster(int u, real* vec_context){
    int num_word_sense = (int)multi_cluster_emb[u][0], lu, i, j;
    real *sims = (real *) malloc(num_word_sense * sizeof(real));
    real max_sim = -100000.0;
    int k = 0;

    for(i = 0; i < num_word_sense; i++){
        lu = i * (dim + 1) + 2;
        real sum, sum1, sum2;
        sum = sum1 = sum2 = 0;
        
        for(j = 0; j < dim; j++){
            sum += vec_context[j] * multi_cluster_emb[u][lu + j];
            sum1 += pow(vec_context[j], 2);
            sum2 += pow(multi_cluster_emb[u][lu + j], 2);
        }

        //cout << multi_cluster_emb[u][lu] << endl;
        //cout << multi_cluster_emb[u][lu + 1] << endl;
        //cin.get();

        sum1 = sqrt(sum1); sum2 = sqrt(sum2);
        sims[i] = sum / (sum1 * sum2 + 1e-8);
        //cout << sum1 << endl; cout << sum2 << endl;
        //cout << sims[i] << endl;
        //cin.get();
        
        if(sims[i] > max_sim){
            k = i;
            max_sim = sims[i]; 
        }
    }
    
    if(num_sense == -1 && max_sim < gap){
        //cout << max_sim << endl;
        //cout << gap << endl;
        //cin.get();
		k = num_word_sense;
		if (k >= max_num_sense - 1)
			k = 0;
        else
			CreatNewClusterV2(u);
    }
    free(sims);
    return k;
    // return AssignSenseBySample(sims);
}

void UpdateContextCluster(int u, int k, real *vec_context)
{
	int lu = k * (dim + 1) + 1;
	int num = multi_cluster_emb[u][lu], i;
	for (i = 0; i < dim; i++){
		multi_cluster_emb[u][lu + 1 + i] = multi_cluster_emb[u][lu + 1 + i] * num / (num + 1) + vec_context[i] / (num + 1);
	}

	multi_cluster_emb[u][lu] += 1;
}

int AssignSense2Node(int u, int v)
{
    real *vec_context = (real *)calloc(dim, sizeof(real));
    int k;
    AssignContextVec(v, vec_context);
    k = FindNearestCluster(u, vec_context);
    UpdateContextCluster(u, k, vec_context);

	free(vec_context);
    return k;
}

void *TrainLINEThread(void *id)
{
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
    int k = 0;
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count>10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

        // 
        k = AssignSense2Node(u, v);
		//printf("word %lld; sense %d\n", u, k);
        //UpdateContextCluster(u, k);
        
		lu = k * dim; // 
		for (int c = 0; c != dim; c++) vec_error[c] = 0;

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target * dim;

			if (order == 2) Update(&multi_sense_emb[u][lu], &emb_context[lv], vec_error, label);
		}
		for (int c = 0; c != dim; c++) multi_sense_emb[u][c + lu] += vec_error[c];

		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

void Output()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
        int num = (int) multi_cluster_emb[a][0];
		if (is_binary) 
        {
            for (int k = 0; k < num; k++){
                fprintf(fo, "%s %d %d ", vertex[a].name, k, (int)multi_cluster_emb[a][(1 + dim)*k + 1]);
                for (int b = 0; b < dim; b++) 
                    fwrite(&multi_sense_emb[a][k * dim + b], sizeof(real), 1, fo);
                fprintf(fo, "\n");
            }
        }
		else {
            for (int k = 0; k < num; k++){
                fprintf(fo, "%s %d %d ", vertex[a].name, k, (int)multi_cluster_emb[a][(1 + dim)*k + 1]);
                for (int b = 0; b < dim; b++) 
					fprintf(fo, "%lf ", multi_sense_emb[a][k * dim + b]);
                fprintf(fo, "\n");
            }
        }
	}
	fclose(fo);
    
    char embedding_file2[100];
    sprintf(embedding_file2, "%s.cemb", embedding_file);
    
    fo = fopen(embedding_file2, "wb");
    fprintf(fo, "%d %d\n", num_vertices, dim);

    for (int a = 0; a < num_vertices; a++)
    {
        fprintf(fo, "%s ", vertex[a].name);
        if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_context[a * dim + b], sizeof(real), 1, fo);
        else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_context[a * dim + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);

    char embedding_file3[100];
    sprintf(embedding_file3, "%s.all.emb", embedding_file);

    fo = fopen(embedding_file3, "wb");

    fprintf(fo, "%d %d\n", num_vertices, dim);
    for(int a = 0; a < num_vertices; a++){
        int num = multi_cluster_emb[a][0];
        real vec[dim];

        int token_num = 0;

        for (int b = 0; b < dim; b++)
            vec[b] = 0;
        
        for (int k = 0; k < num; k++){
            int num_sense = (int) multi_cluster_emb[a][k * (1 + dim) + 1];
            //num_sense = 1;
            token_num += num_sense;
            
            for (int b = 0; b < dim; b++){
                vec[b] += multi_sense_emb[a][k * dim + b] * num_sense;
            }
        }

        if(token_num != 0){
            for (int b = 0; b < dim; b++){
                vec[b] = vec[b] / (1.0 * token_num);
            }
        }
        
        fprintf(fo, "%s ", vertex[a].name);
        if (is_binary) for(int b = 0; b < dim; b++) fwrite(&vec[b], sizeof(real), 1, fo);
        else for(int b = 0; b < dim; b++) fprintf(fo, "%lf ", vec[b]);
        fprintf(fo, "\n");
    }

    fclose(fo);

    char embedding_file4[100];
    sprintf(embedding_file4, "%s.multi.context.emb", embedding_file);

    fo = fopen(embedding_file4, "wb");

    fprintf(fo, "%d %d\n", num_vertices, dim);
    for(int a = 0; a < num_vertices; a++){
        int num = (int) multi_cluster_emb[a][0];
        if (is_binary)
        {
            for (int k = 0; k < num; k++){
                fprintf(fo, "%s %d %d ", vertex[a].name, k, (int)multi_cluster_emb[a][(1 + dim)*k + 1]);
                for (int b = 0; b < dim; b++)
                    fwrite(&multi_cluster_emb[a][k *(dim + 1) + 2 + b], sizeof(real), 1, fo);
                fprintf(fo, "\n");
            }
        }
        else {
            for (int k = 0; k < num; k++){
                fprintf(fo, "%s %d %d ", vertex[a].name, k, (int)multi_cluster_emb[a][(1 + dim)*k + 1]);
                for (int b = 0; b < dim; b++)
                    fprintf(fo, "%lf ", multi_cluster_emb[a][k * (dim + 1) + 2 + b]);
                fprintf(fo, "\n");
            }
        }

    }

    fclose(fo);

}

void TrainLINE() {
	long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	if (order != 1 && order != 2)
	{
		printf("Error: order should be eighther 1 or 2!\n");
		exit(1);
	}
	printf("--------------------------------\n");
	printf("Order: %d\n", order);
    printf("Sense: %d\n", num_sense);
    printf("Gapval: %lf\n", gap);
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

	InitHashTable();
	ReadData();
	InitAliasTable();
	InitVector();
	InitNegTable();
	InitSigmoidTable();

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	Output();
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("MS-LINE: Multi Sense Large Information Network Embedding\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse network data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the learnt embeddings\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-order <int>\n");
		printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-sense <int>\n");
		printf("\t\tSet the fixed number of sense; default is -1 which means non-parameteric estimation");
        printf("\t-gap <float>\n");
        printf("\t\tSet the gab value; default is -0.5");
		printf("\nExamples:\n");
		printf("./ms-line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20 -sense 3\n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sense", argc, argv)) > 0) num_sense = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-gap", argc, argv)) > 0) gap = atof(argv[i + 1]);
    total_samples *= 1000000;
	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainLINE();
	return 0;
}
