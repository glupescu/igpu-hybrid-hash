#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <sstream>
#include <string>
#include <string>
#include <unordered_map>

#include "hash.h"

using namespace std;

/******************************************/

void StdHash::reserve(int size) {
	hash.reserve(size);
}

void StdHash::insert(int key, int value) {
	hash.insert(hash_t::value_type(key, value));
}

int StdHash::get(int key) {
	return hash.find(key)->second;
}

void StdHash::remove(int key) {
	hash.erase(key);
}

void StdHash::insert_batch(int* keys, int* values, int num) {
	for (int i = 0; i < num; i++) { 
		hash.insert(hash_t::value_type(keys[i], values[i]));
	}
}

int* StdHash::get_batch(int* keys, int num) {
	int* retValues = new int[num];
	if (retValues == NULL) {
		return NULL;
	}

	for (int i = 0; i < num; i++) {
		retValues[i] = hash.find(keys[i])->second;
	}

	return retValues;
}

float StdHash::get_load_factor(void) {
		return hash.load_factor();
}
