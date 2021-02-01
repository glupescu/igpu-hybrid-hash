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

#include "hash.h"

using namespace cl::sycl;
using namespace std;

int hashit_1(long long int data,
	long long int limit) {
	return (data * 653267llu) % 17399181177241llu % limit;
}

int hashit_2(long long int data,
	long long int limit) {
	return (data * 1646237llu) % 34798362354533llu % limit;
}

HybridHash::HybridHash() {

	sycl::gpu_selector dev_gpu;
	sycl::cpu_selector dev_cpu;

	try {
		this->bdata = nullptr;

		this->q_gpu = new queue(dev_gpu);
		this->q_cpu = new queue(dev_cpu);

		cout << "Running on hybrid [ iGPU: "
			<< this->q_gpu->get_device().get_info<info::device::name>() << ", ";
		cout << " CPU: " << this->q_cpu->get_device().get_info<info::device::name>() << " ]\n";

		cout << "SLM memory: " << q_gpu->get_device().get_info<sycl::info::device::local_mem_size>() << "\n";

		// allocate memory
		this->reserve();

		this->print();

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
	cout << " --- HASH --- " << std::endl;
	for (int i = 0; i < this->bnum; i++) {
		cout << "\t" << this->bdata[i].keys[0] << " : " << this->bdata[i].values[0]
			<< "\t" << this->bdata[i].keys[1] << " : " << this->bdata[i].values[1]
			<< "\t" << this->bdata[i].keys[2] << " : " << this->bdata[i].values[2]
			<< "\t" << this->bdata[i].keys[3] << " : " << this->bdata[i].values[3] << " \n";
	}
	cout << " -------------------- " << std::endl;
}

void HybridHash::reserve(int bnum) {
	this->bnum = bnum;
	this->bdata = malloc_shared<Bucket>(bnum, *this->q_gpu);

	sycl::buffer<Bucket, 1> bdata_gpu(this->bdata, bnum);

	this->q_gpu->submit([&](sycl::handler& cgh) {
		auto bdata_rw = bdata_gpu.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for(bnum, [=](auto i) {
			bdata_rw[i].keys[0] = -1;
			bdata_rw[i].keys[1] = -1;
			bdata_rw[i].keys[2] = -1;
			bdata_rw[i].keys[3] = -1;
			bdata_rw[i].values[0] = 0;
			bdata_rw[i].values[1] = 0;
			bdata_rw[i].values[2] = 0;
			bdata_rw[i].values[3] = 0;
			});
		});

	this->q_gpu->wait();
}

void HybridHash::insert(int key, int value) {
}
int HybridHash::get(int key) { return 0; }
void HybridHash::remove(int key) {}

void HybridHash::insert_batch(int* keys, int* values, int num) {

	int bnum = this->bnum;
	Bucket* bdata = this->bdata;

	sycl::buffer<int, 1> gpu_keys(keys, range<1>(num));
	sycl::buffer<int, 1> gpu_values(values, range<1>(num));
	sycl::buffer<Bucket, 1> gpu_bdata(bdata, range<1>(bnum));

	this->q_gpu->submit([&](sycl::handler& cgh) {
		auto gpu_keys_r = gpu_keys.get_access<sycl::access::mode::read>(cgh);
		auto gpu_values_r = gpu_values.get_access<sycl::access::mode::read>(cgh);
		auto gpu_bdata_rw = gpu_bdata.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for(range<1>(num), [=](auto i) {
			int gpu_key = gpu_keys_r[i];
			int index1 = hashit_1(gpu_key, bnum);
			int index2 = hashit_2(gpu_key, bnum);

			/*for (int h = 0; h < 4; h++) {
				if (gpu_bdata_rw[index1].keys[h] == -1) {
					gpu_bdata_rw[index1].keys[h] = gpu_key;
					gpu_bdata_rw[index1].values[h] = gpu_values_r[i];
				}
			}*/

			bdata[i % bnum].keys[0] = 3;
			gpu_bdata_rw[i % bnum].keys[0] = 4;
			});
		});

	this->q_gpu->wait();

	this->print();

	cout << "END" << std::endl;
}

int* HybridHash::get_batch(int* keys, int num) { return nullptr; }
float HybridHash::get_load_factor(void) { return 0; }