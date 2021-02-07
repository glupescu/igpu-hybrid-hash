#pragma once
#include <CL/sycl.hpp>
#include <unordered_map>

class Hash {
public:
	virtual bool insert(int key, int value) = 0;
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
	bool insert(int key, int value);

	int get(int key);
	void remove(int key);

	void insert_batch(int* keys, int* values, int num);
	int* get_batch(int* keys, int num);
	float get_load_factor(void);
};

#define BUCKET_SIZE		4

union Item {
	uint64_t raw;

	struct {
		uint32_t key;
		uint32_t value;
	} data;

	Item() {
		this->data.key = -1;
		this->data.value = 0;
	}

	Item(int key, int value) {
		this->data.key = key;
		this->data.value = value;
	}
};

struct Work {

	Work(uint64_t* items, int num) {
		this->items = items;
		this->num = num;
	}

	uint64_t* items;
	int num;
};

class HybridHash : Hash {

public:
	HybridHash();

	sycl::queue *q_gpu;
	sycl::queue *q_cpu;

	Item*	idata;
	sycl::atomic<bool>* idata_at;
	int		inum;

	float c_lfactor;
	float t_lfactor;

	void print(void);

	void reserve(int bnum = 10);
	bool insert(int key, int value);
	bool insert(uint64_t item, int lvl = 0);

	void remove(int key);

	void insert_batch(int* keys, int* values, int num);

	Work insert_gpu(Work work);
	Work insert_cpu(Work work);

	int get(int key);
	int* get_batch(int* keys, int num);
	int* get_batch2(int* keys, int num);
	float get_load_factor(void);
};