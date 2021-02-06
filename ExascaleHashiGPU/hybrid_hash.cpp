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

#include "hash.h"

using namespace cl::sycl;
using namespace std;

int hashit_1(long long int data,
	long long int limit) {
	return (data * 12289llu) % 98317llu % limit;
}

int hashit_2(long long int data,
	long long int limit) {
	return (data * 24593llu ) % 196613llu % limit;
}

int hashit_3(long long int data,
	long long int limit) {
	return (data * 467237u) % 1162687llu % limit;
}

int hashit_4(long long int data,
	long long int limit) {
	return (data * 90523llu) % 807403llu % limit;
}

HybridHash::HybridHash() {

	sycl::gpu_selector dev_gpu;
	sycl::cpu_selector dev_cpu;

	try {
		this->idata = nullptr;

		this->q_gpu = new queue(dev_gpu);
		this->q_cpu = new queue(dev_cpu);

		cout << "Running on hybrid [ iGPU: "
			<< this->q_gpu->get_device().get_info<info::device::name>() << ", ";
		cout << " CPU: " << this->q_cpu->get_device().get_info<info::device::name>() << " ]\n";

		cout << "SLM memory: " << q_gpu->get_device().get_info<sycl::info::device::local_mem_size>() << "\n";

		// allocate memory
		this->reserve();

		//this->print();

		//range num_items{ array_size };

		//auto e = q_gpu.parallel_for(num_items, [=](auto i) { array2[i] = f(i); array[i] = f(i); });

		// q.parallel_for() is an asynchronous call. DPC++ runtime enqueues and runs
		// the kernel asynchronously. Wait for the asynchronous call to complete.
		//e.wait();

		//cout << array[1000] << std::endl;

		//free(array2, q_gpu);
		//free(array, q_gpu);
	}
	catch (std::exception const& e) {
		cout << "An exception is caught while computing on device.\n";
		terminate();
	}

	cout << "success" << std::endl;
}

void HybridHash::print(void) {

	std::unordered_map<int, int> histogram;
	int freeItems = 0;

	cout << std::endl << " --- HASH --- " << std::endl;
	for (int i = 0; i < this->inum; i++) {

		Item item = this->idata[i];

		if (item.data.key != -1) {
			std::cout << std::setw(4) << "[" << std::setw(8) << item.data.key << ":" << std::setw(8) << item.data.value << " ]";
		}
		else {
			freeItems += 1;
			std::cout << std::setw(4) << "[" << std::setw(8) << "******" << ":" << std::setw(8) << "******" << " ]";
			//std::cout << "\t[ \t ***** : **** ]";
		}

		if (i % BUCKET_SIZE == 0) {
			std::cout << std::endl;
		}
	}

	cout << "FREE: " << freeItems << " MAX:" << this->inum << std::endl;

	/*
	std::cout << "RES" << std::endl;
	for (auto const& x : histogram) {
		std::cout << x.first << " : " << x.second << std::endl;
	}
			cout << "\t" << this->bdata[i].keys[0] << " : " << this->bdata[i].values[0] << " \n";
			//<< "\t" << this->bdata[i].keys[1] << " : " << this->bdata[i].values[1]
			//<< "\t" << this->bdata[i].keys[2] << " : " << this->bdata[i].values[2]
			//<< "\t" << this->bdata[i].keys[3] << " : " << this->bdata[i].values[3] << " \n";
	}
	cout << " -------------------- " << std::endl;
	
	*/
}

void HybridHash::reserve(int inum) {
	this->inum = inum;
	this->idata = malloc_shared<Item>(inum, *this->q_gpu);
	auto idata_gpu = this->idata;
	//this->idata_at = malloc_device<sycl::atomic<bool>>(inum, *this->q_gpu);

	//sycl::buffer<Item, 1> idata_gpu(this->idata, inum);
	//sycl::buffer<sycl::atomic<bool>, 1> idata_at_gpu(this->idata_at, inum);

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

int HybridHash::get(int key) { return 0; }
void HybridHash::remove(int key) {}

Work HybridHash::insert_gpu(Work work) {

	uint64_t* items = work.items;
	int num = work.num;

	this->reserve(num);

	int inum = this->inum;
	uint64_t* idata = (uint64_t*)&this->idata[0].raw;
	//sycl::atomic<bool>* idata_at = this->idata_at;

	sycl::buffer<Item, 1> idata_gpu(this->idata, inum);

	uint64_t* count_not_inserted = malloc_shared<uint64_t>(inum, *this->q_gpu);
	count_not_inserted[0] = 0;

	this->q_gpu->submit([&](sycl::handler& cgh) {

		Item empty;
		empty.data.key = -1;
		empty.data.value = 0;

		cgh.parallel_for(range<1>(num), [=](auto i) {

			Item toInsert;
			toInsert.raw = items[i];

			int index[4];
			index[0] = hashit_1(toInsert.data.key, inum);
			index[1] = hashit_2(toInsert.data.key, inum);
			index[2] = hashit_3(toInsert.data.key, inum);
			index[3] = hashit_4(toInsert.data.key, inum);

			sycl::atomic<uint64_t> atomic_counter{ sycl::global_ptr<uint64_t> {&count_not_inserted[0]} };

			bool foundFree = false;

			for (int hi = 0; hi < 2; hi++) {
				int index1 = index[hi];

				for (int i = index1; (i < index1 + 8) && (i < inum); i++) {
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

			//sycl::atomic<int> atomic_counter{ sycl::global_ptr<int> {&bdata[index1].count} };
			//atomic_counter.fetch_add(1);

			});
		}).wait();

	this->print();

	std::cout << std::endl << "END: " << count_not_inserted[0] << std::endl;
	for (int i = 1; i <= count_not_inserted[0]; i++) {
		Item item;
		item.raw = count_not_inserted[i];
		std::cout << item.data.key << " "; 
	}
	
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

	int index[4];
	index[0] = hashit_1(toInsert.data.key, inum);
	index[1] = hashit_2(toInsert.data.key, inum);
	index[2] = hashit_3(toInsert.data.key, inum);
	index[3] = hashit_4(toInsert.data.key, inum);

	bool foundFree = false;

	for (int hi = 0; hi < 2; hi++) {
		int index1 = index[hi];
		for (int i = index1; (i < index1 + 4) && (i < inum); i++) {
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

	uint64_t* items = malloc_shared<uint64_t>(inum, *this->q_gpu);
	for (int i; i < num; i++) {
		Item item;
		item.data.key = keys[i];
		item.data.value = values[i];

		items[i] = item.raw;
	}

	this->insert_cpu(this->insert_gpu(Work(items, num)));

	this->print();
}

int* HybridHash::get_batch(int* keys, int num) { return nullptr; }
float HybridHash::get_load_factor(void) { return 0; }