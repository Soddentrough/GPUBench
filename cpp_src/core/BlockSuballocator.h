#pragma once

#include <cstdint>
#include <vector>
#include <mutex>

namespace gpubench {

struct MemoryChunk {
  uint64_t offset = 0;
  uint64_t size = 0;
  bool inUse = false;
};

struct MemoryBlock {
  uint32_t blockIndex = 0;
  uint64_t size = 0;
  uint32_t memoryTypeIndex = 0;
  bool hasDeviceAddress = false;
  std::vector<MemoryChunk> chunks;
};

struct SuballocResult {
  bool success = false;
  uint32_t blockIndex = 0;
  uint64_t offset = 0;
  bool requiresNewBlock = false;
};

class BlockSuballocator {
public:
  static constexpr uint64_t kDefaultBlockSize = 64ULL * 1024ULL * 1024ULL;
  static constexpr uint64_t kMaxSuballocSize = 32ULL * 1024ULL * 1024ULL;

  BlockSuballocator() = default;
  ~BlockSuballocator() = default;

  // Align an offset up to the nearest multiple of alignment
  static inline uint64_t alignUp(uint64_t offset, uint64_t alignment) {
    if (alignment <= 1) return offset;
    return (offset + alignment - 1) & ~(alignment - 1);
  }

  // Attempt to allocate from existing blocks.
  // Returns success=true if allocated, or requiresNewBlock=true if no existing block has capacity.
  SuballocResult allocate(uint64_t size, uint64_t alignment,
                          uint32_t memoryTypeIndex, bool needDeviceAddress);

  // Register a newly created block of size blockSize and allocate size from it.
  SuballocResult registerNewBlock(uint64_t size, uint64_t blockSize,
                                  uint32_t memoryTypeIndex, bool needDeviceAddress);

  // Free an allocation and coalesce adjacent free chunks
  bool free(uint32_t blockIndex, uint64_t offset);

  // Clear all blocks
  void clear();

  // Inspect blocks for testing and verification
  const std::vector<MemoryBlock> &getBlocks() const { return m_blocks; }

private:
  mutable std::mutex m_mutex;
  std::vector<MemoryBlock> m_blocks;
};

} // namespace gpubench
