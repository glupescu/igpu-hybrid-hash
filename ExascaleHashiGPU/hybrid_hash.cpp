//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif
#include <array>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>

#include "hash.h"

#define DEBUG	0

using namespace cl::sycl;
using namespace std;

int hashit_1(uint64_t data,
	uint64_t limit) {
	return (data * 653267llu) % 17399181177241llu % limit;
}

int hashit_2(uint64_t data,
	uint64_t limit) {
	return (data * 205759llu) % 278386898836457llu % limit;
}

int hashit_3(uint64_t data,
	uint64_t limit) {
	return (data * 1646237llu) % 34798362354533llu % limit;
}

int hashit_4(uint64_t data,
	uint64_t limit) {
	return (data * 90523llu) % 807403llu % limit;
}

HybridHash::HybridHash() {

	sycl::gpu_selector dev_gpu;
	sycl::cpu_selector dev_cpu;

	this->t_lfactor = 0.9f;
	this->c_lfactor = 0;

	try {
		this->idata = nullptr;

		this->q_gpu = new queue(dev_gpu);
		this->q_cpu = new queue(dev_cpu);

		cout << "Running on hybrid [ iGPU: "
			<< this->q_gpu->get_device().get_info<info::device::name>() << ", ";
		cout << " CPU: " << this->q_cpu->get_device().get_info<info::device::name>() << " ]\n";

		cout << "SLM memory: " << q_gpu->get_device().get_info<sycl::info::device::local_mem_size>() << "\n";

		this->reserve();
	}
	catch (std::exception const& e) {
		cout << "An exception is caught while computing on device.\n";
		terminate();
	}

	cout << "success" << std::endl;
}

void HybridHash::print(void) {

	if (!DEBUG) {
		return;
	}

	std::unordered_map<int, int> histogram;
	int freeItems = 0;

	cout << std::endl << " --------- HASH ---------- " << std::endl;

	int num_sampled = 80;

	// start with fixed interval and randomly select afterwards
	int start_sample = 0;
	int end_sample = std::min(this->inum, num_sampled);

	if (this->inum > num_sampled) {
		std::random_device rd;
		std::mt19937 gen(rd());
		uniform_int_distribution<uint32_t> inum_random(1, this->inum - num_sampled);

		start_sample = inum_random(gen);
		end_sample = start_sample + num_sampled;
	}

	for (int i = start_sample; i < end_sample; i++) {

		Item item = this->idata[i];

		if (item.data.key != -1) {
			std::cout << std::setw(4) << "[" << std::setw(8) << item.data.key << ":" << std::setw(8) << item.data.value << " ]";
		}
		else {
			freeItems += 1;
			std::cout << std::setw(4) << "[" << std::setw(8) << "******" << ":" << std::setw(8) << "******" << " ]";
		}

		if (i % BUCKET_SIZE == 0) {
			std::cout << std::endl;
		}
	}

	cout << " ====> LOAD FACTOR: " << (this->get_load_factor() * 100) << "%" << std::endl;
}

void HybridHash::reserve(int inum) {
	this->inum = inum;
	this->idata = malloc_shared<Item>(inum, *this->q_gpu);
	auto idata_gpu = this->idata;
	//this->idata_at = malloc_device<sycl::atomic<bool>>(inum, *this->q_gpu);

	//sycl::buffer<Item, 1> idata_gpu(this->idata, inum);
	//sycl::buffer<sycl::atomic<bool>, 1> idata_at_gpu(this->idata_at, inum);

	if (this->idata == nullptr) {
		cout << "Could not allocate memory" << std::endl;
		exit(1);
	}

	this->q_gpu->submit([&](sycl::handler& cgh) {
		//auto idata_rw = idata_gpu.get_access<sycl::access::mode::read_write>(cgh);
		//auto idatw_at = idata_gpu.get_access<sycl::access::mode::atomic>(cgh);

		//auto idata_at_gpu_a = idata_at_gpu.get_access<sycl::access::mode::atomic > (cgh);

		cgh.parallel_for(inum, [=](auto i) {
			idata_gpu[i].data.key = -1;
			idata_gpu[i].data.value = 0;
			//idata_at_gpu_a[i] = false;
			});
		});

	this->q_gpu->wait();

	cout << "Values stored" << std::endl;

	this->print();
}

void HybridHash::remove(int key) {}

Work HybridHash::insert_gpu(Work work) {

	uint64_t* items = work.items;
	int num = work.num;

	int inum = this->inum;
	uint64_t* idata = (uint64_t*)&this->idata[0].raw;
	//sycl::atomic<bool>* idata_at = this->idata_at;

	uint64_t* count_not_inserted = malloc_shared<uint64_t>(inum, *this->q_gpu);
	count_not_inserted[0] = 0;

	this->q_gpu->submit([&](sycl::handler& cgh) {

		Item empty;
		empty.data.key = -1;
		empty.data.value = 0;

		cgh.parallel_for(range<1>(num), [=](auto i) {

			Item toInsert;
			toInsert.raw = items[i];

			int index[3];
			index[0] = hashit_1(toInsert.data.key, inum);
			index[1] = hashit_2(toInsert.data.key, inum);
			index[2] = hashit_3(toInsert.data.key, inum);

			sycl::atomic<uint64_t> atomic_counter{ sycl::global_ptr<uint64_t> {&count_not_inserted[0]} };

			bool foundFree = false;

			for (int hi = 0; hi < 3; hi++) {
				int index1 = index[hi];

				for (int i = index1; (i < index1 + 64) && (i < inum); i++) {
					sycl::atomic<uint64_t> idata_gpu_at{ sycl::global_ptr<uint64_t> {&idata[i]} };
					if (idata_gpu_at.compare_exchange_strong((uint64_t&)empty.raw, toInsert.raw)) {
						foundFree = true;
						break;
					}
				}

				if (foundFree) {
					break;
				}
			}

			if (!foundFree) {
				uint64_t idx = atomic_counter.fetch_add(1) + 1;
				count_not_inserted[idx] = toInsert.raw;
			}

			});
		}).wait();

	this->print();

	std::cout << std::endl << "END: " << count_not_inserted[0] << std::endl;
	
	return Work(&count_not_inserted[1], count_not_inserted[0]);
}

bool HybridHash::insert(int key, int value) {
	Item item;
	item.data.key = key;
	item.data.value = value;

	return insert(item.raw);
}

bool HybridHash::insert(uint64_t item, int lvl) {
	int inum = this->inum;

	Item toInsert;
	toInsert.raw = item;

	if (lvl > 16) {
		//cout << "LEVEL STACK FAILED !!!! " << toInsert.data.key << std::endl;
		return false;
	}

	Item empty;
	empty.data.key = -1;
	empty.data.value = 0;

	int index[3];
	index[0] = hashit_1(toInsert.data.key, inum);
	index[1] = hashit_2(toInsert.data.key, inum);
	index[2] = hashit_3(toInsert.data.key, inum);

	bool foundFree = false;

	for (int hi = 0; hi < 3; hi++) {
		int index1 = index[hi];
		for (int i = index1; (i < index1 + 64) && (i < inum); i++) {
			if (this->idata[i].raw == empty.raw) {
				this->idata[i].raw = item;
				return true;
			}
		}
	}

	if (this->insert(this->idata[ index[0] ].raw, lvl + 1) == true) {
		this->idata[index[0]].raw = item;
		return true;
	}
	else {
		//cout << "F -----> " << toInsert.data.key << std::endl;
		return false;
	}
}


Work HybridHash::insert_cpu(Work work) {
	uint64_t* items = work.items;
	int num = work.num;

	cout << "CPU LEFT TO WORK " << num << std::endl;
	// cuckoo

	int inserted = 0;
	for (int i = 0; i < num; i++) {
		if (this->insert(work.items[i]) == true) {
			inserted += 1;
		}
	}

	cout << "CPU LEFT TO WORK " << num - inserted << std::endl;

	return work;
}

void HybridHash::insert_batch(int* keys, int* values, int num) {

	// target load factor 90%
	int resize = (this->inum * this->get_load_factor() + num) * 1 / this->t_lfactor + 1000;

	this->reserve(resize);

	uint64_t* items = malloc_shared<uint64_t>(num, *this->q_gpu);
	for (int i = 0; i < num; i++) {
		items[i] = Item(keys[i], values[i]).raw;
	}

	this->insert_cpu(this->insert_gpu(Work(items, num)));

	this->print();
}

int HybridHash::get(int key) {
	int index[3];
	index[0] = hashit_1(key, inum);
	index[1] = hashit_2(key, inum);
	index[2] = hashit_3(key, inum);

	bool hasFound = false;
	for (int hi = 0; hi < 3; hi++) {
		int index1 = index[hi];

		for (int i = index1; (i < index1 + 64) && (i < inum); i++) {
			if (idata[i].data.key == key) {
				return idata[i].data.value;
			}
		}
	}

	return 0;
}

int* HybridHash::get_batch2(int* keys, int num) {
	int* values = new int[num];

	for (int i = 0; i < num; i++)
		values[i] = this->get(keys[i]);

	return values;
}

int* HybridHash::get_batch(int* keys, int num) {

	// alloc shared memory
	int* values = malloc_shared<int>(num, *this->q_gpu);
	int inum = this->inum;
	Item* gpu_idata = this->idata;
	
	sycl::buffer<int, 1> idata_keys(keys, num);

	this->q_gpu->submit([&](sycl::handler& cgh) {

		auto gpu_idata_keys = idata_keys.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for(range<1>(num), [=](auto i) {

			int keyToSearch = gpu_idata_keys[i];

			int index[3];
			index[0] = hashit_1(keyToSearch, inum);
			index[1] = hashit_2(keyToSearch, inum);
			index[2] = hashit_3(keyToSearch, inum);

			bool hasFound = false;
			for (int hi = 0; hi < 3; hi++) {
				int index1 = index[hi];

				for (int ih = index1; (ih < index1 + 64) && (ih < inum); ih++) {
					if (gpu_idata[ih].data.key == keyToSearch) {
						values[i] = gpu_idata[ih].data.value;
						hasFound = true;
						break;
					}
				}

				if (hasFound) {
					break;
				}
			}

			if (hasFound == false) {
				values[i] = 0;
			}
			});
	}).wait();

	return values;
}

float HybridHash::get_load_factor(void) {

	int count = 0;

	for (int i = 0; i < this->inum; i++) {
		// check if not empty
		if (this->idata[i].raw != Item().raw) {
			count++;
		}
	}

	return (float) count / (float) this->inum; 
}