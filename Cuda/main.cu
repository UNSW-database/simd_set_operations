#include <algorithm>
#include <iostream>
#include <span>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include <memory>

struct range {
	size_t start;
	size_t end;
};
template <typename T>
__global__ void cuda_warp_gather(
	T* mainData, range* mainRanges, size_t mainSetCount,
	T* outputData, range* outputRanges, size_t* outputSetsCount) {
	T idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x + 1 == gridDim.x and threadIdx.x == 0 and idx + gridDim.x == mainSetCount) {
		for (auto i = mainRanges[blockIdx.x].start; i < mainRanges[blockIdx.x].end; ++i) {
			outputData[outputRanges[blockIdx.x].end] = mainData[i];
			++outputRanges[blockIdx.x].end;
		}
		return;
	}
	if (idx + gridDim.x >= mainSetCount || blockIdx.x >= mainSetCount) return;
	__shared__ size_t mainIndex;
	__shared__ size_t mainEnd;
	if (threadIdx.x == 0 and blockIdx.x == 0) {
	*outputSetsCount = gridDim.x;
	}
	if (threadIdx.x == 0) {
		mainIndex = mainRanges[blockIdx.x].start;
		mainEnd = mainRanges[blockIdx.x].end;
	}
	__syncthreads();
	size_t currIndex = mainRanges[idx+gridDim.x].start;
	size_t end = mainRanges[idx+gridDim.x].end;
	while (true) {
		T value = mainData[mainIndex];
		while (currIndex < end && mainData[currIndex] < value) {
			++currIndex;
		}
		bool withinRange = currIndex < end;
		auto mask = __activemask();
		bool withinAll = __all_sync(mask, withinRange);
		// std::cout << "Index " << currIndex << " vs " << mainIndex << std::endl;
		if (!withinAll || mainIndex >= mainEnd) {
			return;
		}

		bool eq = mainData[currIndex] == value;
		mask = __activemask();
		if (__all_sync(mask, eq)) {
			__syncthreads();
			if (threadIdx.x == 0) {
				outputData[outputRanges[blockIdx.x].end] = value;
				++outputRanges[blockIdx.x].end;
				++mainIndex;
			}
			__syncthreads();
			++currIndex;
		} else {
			if (threadIdx.x == 0) {
				++mainIndex;
			}
		}
	}
}

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
template <typename T, typename Container>
std::pair<std::vector<T>, std::vector<range>> flatten(Container input) {
	std::pair<std::vector<T>, std::vector<range>> output;
	size_t i = 0;
	for (auto& inner: input) {
		output.second.push_back({i, i + inner.size()});
		output.first.insert(output.first.end(), inner.begin(), inner.end());
		i += inner.size();
	}
	return output;
}
struct intSpan {
	int* ptr;
	size_t size_;
	auto empty() const {
		return size() == 0;
	}
	auto data() const {
		return ptr;
	}
	auto size() const -> size_t {
		return size_;
	}
	auto begin() const {
		return ptr;
	}
	auto end() const {
		return ptr + size_;
	}
	const auto cbegin() const {
		return ptr;
	}
	const auto cend() const {
		return ptr + size_;
	}
};
struct rangeSpan {
	range* ptr;
	size_t size_;
	auto empty() const {
		return size() == 0;
	}
	auto data() const {
		return ptr;
	}
	auto size() const -> size_t {
		return size_;
	}
	auto begin() const {
		return ptr;
	}
	auto end() const {
		return ptr + size_;
	}
	const auto cbegin() const {
		return ptr;
	}
	const auto cend() const {
		return ptr + size_;
	}
};
struct dataAndRange {
	// These are named first and second to maintain template compatibility with an std::pair
	intSpan first;
	rangeSpan second;
	auto getData() const {return first;}
	auto getRanges() const {return second;}
};
template <typename T, typename Storage>
// std::pair<std::vector<T>, std::vector<range>>
Storage& cudaGatherWrapper(Storage& input, Storage& output) {
	auto& main = input;
	auto SIZE = main.second.size();
	int blockCount = std::ceil(SIZE/static_cast<double>(33));
	auto biggestRange = *std::max_element(main.second.begin(), main.second.end(), [](const auto& a, const auto& b) {
		return a.end - a.start < b.end - b.start;
	});
	auto maxSetSize = biggestRange.end - biggestRange.start;
	T* data1; range* ranges1; size_t data1Size;
	T* data2; range* ranges2; size_t data2Size;
	T* outputDataBase; range* outputRangesBase; size_t outputSetsCount;
	auto mal1 = cudaMalloc(&data1, main.first.size()*sizeof(T));
	auto mem1 = cudaMemcpy(data1, main.first.data(), main.first.size()*sizeof(T), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	auto mal2 = cudaMalloc(&ranges1, main.second.size()*sizeof(range));
	auto mem2 = cudaMemcpy(ranges1, main.second.data(), main.second.size()*sizeof(range), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	auto mal5 = cudaMalloc(&outputDataBase, sizeof(T)*maxSetSize*(blockCount));
	auto mal6 = cudaMalloc(&outputRangesBase, sizeof(range)*(blockCount));
	cudaDeviceSynchronize();
	std::vector<range> outputInitial;
	for (size_t i = 0; i < blockCount; i++) {
		outputInitial.push_back({i*maxSetSize, i*maxSetSize});
	}
	auto mem6 = cudaMemcpy(outputRangesBase, outputInitial.data(), outputInitial.size()*sizeof(range), cudaMemcpyHostToDevice);
	cudaMalloc(&data2, blockCount*maxSetSize*sizeof(T));
	cudaMalloc(&ranges2, outputInitial.size()*sizeof(range));
	cudaMemcpy(ranges2, outputRangesBase, sizeof(range)*blockCount, cudaMemcpyDeviceToDevice);
	size_t* outputSize;
	cudaMalloc(&outputSize, sizeof(size_t));
	cudaDeviceSynchronize();
	size_t cpuOutputSize;
	do {
		cuda_warp_gather<<<blockCount, 32>>>(data1, ranges1, SIZE, data2, ranges2, outputSize);
		cudaDeviceSynchronize();
		cudaMemcpy(&cpuOutputSize, outputSize, sizeof(size_t), cudaMemcpyDeviceToHost);
		std::swap(data1, data2);
		std::swap(ranges1, ranges2);
		cudaMemcpy(ranges2, outputRangesBase, cpuOutputSize*sizeof(range), cudaMemcpyDeviceToDevice);
		SIZE = cpuOutputSize;
		blockCount = std::ceil(SIZE/static_cast<double>(33));
	} while (cpuOutputSize > 1);
	auto& cpuOutputRanges = output.second;
	auto& cpuOutputData = output.first;
	cudaDeviceSynchronize();
	auto val = cudaMemcpy(cpuOutputRanges.data(), ranges1, cpuOutputSize*sizeof(range), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaMemcpy(cpuOutputData.data(), data1, cpuOutputData.size()*sizeof(T), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(data1);
	cudaFree(data2);
	cudaFree(ranges1);
	cudaFree(ranges2);
	cudaFree(outputDataBase);
	cudaFree(outputRangesBase);
	cudaFree(outputSize);
	return output;
}
extern "C" void cudaGatherWrapperC(dataAndRange input, dataAndRange output) {
	cudaGatherWrapper<int>(input, output);
}

#ifndef BUILDING_RUST_LIB
int main(int argc, char** argv) {
	size_t SIZE;
	if (argc < 2) {
		SIZE = 100;
	}
	else {
		SIZE = atoll(argv[1]);
	}
	const size_t maxSetSize = 100;
	std::vector<int> dataToCopy;
	dataToCopy.reserve(maxSetSize);
	for (int i = 0; i < maxSetSize; i++) {
		dataToCopy.push_back(i+1);
	}
	std::vector<std::vector<int>> sets;
	sets.reserve(SIZE);
	sets.push_back(std::vector{2, 3, 4, 5});
	for (int  i = 1; i < SIZE; i++) {
		sets.push_back(dataToCopy);
	}
	auto main = flatten<int>(sets);
	dataAndRange rawData{{main.first.data(), main.first.size()}, {main.second.data(), main.second.size()}};
	decltype(main) initialDataStorage = {std::vector<int>(maxSetSize), {{0,0}}};
	decltype(rawData) initOut{{initialDataStorage.first.data(), initialDataStorage.first.size()}, {initialDataStorage.second.data(), initialDataStorage.second.size()}};

	auto output = cudaGatherWrapper<int>(rawData, initOut);
	// auto output= cudaGatherWrapper<int>(main,initialDataStorage);
	if (output.first.empty()) {
		std::cout << "empty" << std::endl;
	}
	auto it = output.first.begin();
	auto sizeOfOut = output.second.begin()->end - output.second.begin()->start;
	for (auto i = 0; i < sizeOfOut; i++) {
		auto content = *it;
		std::cout << content << " ";
		++it;
	}
	std::cout << std::endl;
	return 0;
}
#endif
