#pragma once
#include <CL/sycl.hpp>
#include <unordered_map>

class Hash {
public:
	virtual void insert(int key, int value) = 0;
	virtual int get(int key) = 0;
	virtual void remove(int key) = 0;
	virtual void reserve(int num) = 0;

	virtual void insert_batch(int* keys, int* values, int num) = 0;
	virtual int* get_batch(int* keys, int num) = 0;

	virtual float get_load_factor(void) = 0;
};

typedef std::unordered_map<int, int, std::hash<int>> hash_t;

class StdHash : Hash {
	hash_t hash;
public:
	void reserve(int size);
	void insert(int key, int value);

	int get(int key);
	void remove(int key);

	void insert_batch(int* keys, int* values, int num);
	int* get_batch(int* keys, int num);
	float get_load_factor(void);
};

#define BUCKET_SIZE		4

struct Bucket {
	int keys[BUCKET_SIZE];
	int values[BUCKET_SIZE];
};

class HybridHash : Hash {

public:
	HybridHash();

	sycl::queue *q_gpu;
	sycl::queue *q_cpu;

	Bucket* bdata;
	int		bnum;

	void print(void);

	void reserve(int bnum = 10);
	void insert(int key, int value);

	int get(int key);
	void remove(int key);

	void insert_batch(int* keys, int* values, int num);
	int* get_batch(int* keys, int num);
	float get_load_factor(void);
};