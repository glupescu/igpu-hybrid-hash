#include <inttypes.h>
#include <sys/types.h>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>
#include <array>
#include <errno.h>
#include <set>

#include "hash.h"

#define	KEY_INVALID		0
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)

using namespace std;

void fillRandom(vector<int>& vecKeys, vector<int>& vecValues, int numEntries) {
	vecKeys.reserve(numEntries);
	vecValues.reserve(numEntries);

	//int interval = (numeric_limits<int>::max() / numEntries) - 1;
	int interval = 0;

	//uniform_int_distribution<uint32_t> distribution(1, INT32_MAX);
	uniform_int_distribution<uint32_t> distribution(1, numEntries * 10);
	set<uint32_t> results;

	// TODO improve
	std::random_device rd;
	std::mt19937 generator(rd());

	while (results.size() != numEntries) {
		results.insert(distribution(generator));
	}

	for (auto &n : results) {
		vecKeys.push_back(n);
		vecValues.push_back(n + 10);
	}

	//shuffle(vecKeys.begin(), vecKeys.end(), g);
	//shuffle(vecValues.begin(), vecValues.end(), g);
}

int main(int argc, char** argv)
{
	clock_t begin;
	double elapsedTime;

	int numKeys = 0;
	int numChunks = 0;
	vector<int> vecKeys;
	vector<int> vecValues;
	int* valuesGot = NULL;

	if (argc == 1) {
		numKeys = 100000;
		numChunks = 1;
	} else {
		DIE(argc != 3,
			"ERR, args num, call ./bin test_numKeys test_numChunks");

		numKeys = stoi(argv[1]);
		DIE((numKeys < 1) || (numKeys >= numeric_limits<int>::max()),
			"ERR, numKeys should be greater or equal to 1 and less than maxint");

		numChunks = stoi(argv[2]);
		DIE((numChunks < 1) || (numChunks >= numKeys),
			"ERR, numChunks should be greater or equal to 1");
	}

	fillRandom(vecKeys, vecValues, numKeys);

	HybridHash* hash = new HybridHash();
	//StdHash* hash = new StdHash();

	int chunkSize = numKeys / numChunks;
	//hash->reserve(chunkSize);

	for (int chunk_start = 0; chunk_start < numKeys; chunk_start += chunkSize) {

		int* keys_start = &vecKeys[chunk_start];
		int* values_start = &vecValues[chunk_start];

		auto begin = chrono::high_resolution_clock::now();

		//cout << "K: ";
		//for (int i = chunk_start; i < chunkSize; i++) {
		//	cout << vecKeys[i] << " ";
		//}
		//cout << std::endl;

		// insert stage
		hash->insert_batch(keys_start, values_start, chunkSize);

		auto end = chrono::high_resolution_clock::now();
		auto elapsedTime = chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

		cout << "HASH_BATCH_INSERT, " << chunkSize
			<< ", " << elapsedTime
			<< "us, " << 100.f * hash->get_load_factor() << endl;
	}

	for (int chunk_start = 0; chunk_start < numKeys; chunk_start += chunkSize) {

		int* keys_start = &vecKeys[chunk_start];

		auto begin = chrono::high_resolution_clock::now();

		// get stage
		valuesGot = hash->get_batch(keys_start, chunkSize);

		auto end = chrono::high_resolution_clock::now();
		auto elapsedTime = chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

		cout << "HASH_BATCH_GET, " << chunkSize
			<< ", " << elapsedTime
			<< "us, " << 100.f * hash->get_load_factor() << endl;

		DIE(valuesGot == NULL, "ERR, ptr valuesCheck cannot be NULL");

		int mistmatches = 0;
		for (int i = 0; i < chunkSize; i++) {
			if (vecValues[chunk_start + i] != valuesGot[i]) {
				mistmatches++;
				if (mistmatches < 32) {
					cout << "Expected " << vecValues[chunk_start + i]
						<< ", but got " << valuesGot[i] << " for key:" << keys_start[i] << endl;
				}
			}
		}

		if (mistmatches > 0) {
			cout << "ERR, mistmatches: " << mistmatches << " / " << numKeys << endl;
			exit(1);
		}
	}

	return 0;
}

