#include "BlockSuballocator.h"

namespace gpubench {

SuballocResult BlockSuballocator::allocate(uint64_t size, uint64_t alignment,
                                           uint32_t memoryTypeIndex,
                                           bool needDeviceAddress) {
  std::lock_guard<std::mutex> lock(m_mutex);

  for (uint32_t b = 0; b < static_cast<uint32_t>(m_blocks.size()); ++b) {
    auto &block = m_blocks[b];
    if (block.memoryTypeIndex != memoryTypeIndex) continue;
    if (block.hasDeviceAddress != needDeviceAddress) continue;

    for (size_t c = 0; c < block.chunks.size(); ++c) {
      if (block.chunks[c].inUse) continue;

      uint64_t chunkStart = block.chunks[c].offset;
      uint64_t chunkSize = block.chunks[c].size;
      uint64_t alignedStart = alignUp(chunkStart, alignment);
      uint64_t padding = alignedStart - chunkStart;

      if (chunkSize >= padding + size) {
        uint64_t remaining = chunkSize - (padding + size);

        if (padding > 0) {
          block.chunks[c].size = padding;

          MemoryChunk allocChunk;
          allocChunk.offset = alignedStart;
          allocChunk.size = size;
          allocChunk.inUse = true;
          block.chunks.insert(block.chunks.begin() + c + 1, allocChunk);

          if (remaining > 0) {
            MemoryChunk remChunk;
            remChunk.offset = alignedStart + size;
            remChunk.size = remaining;
            remChunk.inUse = false;
            block.chunks.insert(block.chunks.begin() + c + 2, remChunk);
          }
        } else {
          block.chunks[c].size = size;
          block.chunks[c].inUse = true;

          if (remaining > 0) {
            MemoryChunk remChunk;
            remChunk.offset = alignedStart + size;
            remChunk.size = remaining;
            remChunk.inUse = false;
            block.chunks.insert(block.chunks.begin() + c + 1, remChunk);
          }
        }

        SuballocResult res;
        res.success = true;
        res.blockIndex = b;
        res.offset = alignedStart;
        res.requiresNewBlock = false;
        return res;
      }
    }
  }

  SuballocResult res;
  res.success = false;
  res.requiresNewBlock = true;
  return res;
}

SuballocResult BlockSuballocator::registerNewBlock(uint64_t size, uint64_t blockSize,
                                                   uint32_t memoryTypeIndex,
                                                   bool needDeviceAddress) {
  std::lock_guard<std::mutex> lock(m_mutex);

  uint32_t newBlockIdx = static_cast<uint32_t>(m_blocks.size());

  MemoryBlock newBlock;
  newBlock.blockIndex = newBlockIdx;
  newBlock.size = blockSize;
  newBlock.memoryTypeIndex = memoryTypeIndex;
  newBlock.hasDeviceAddress = needDeviceAddress;

  MemoryChunk allocChunk;
  allocChunk.offset = 0;
  allocChunk.size = size;
  allocChunk.inUse = true;
  newBlock.chunks.push_back(allocChunk);

  if (blockSize > size) {
    MemoryChunk remChunk;
    remChunk.offset = size;
    remChunk.size = blockSize - size;
    remChunk.inUse = false;
    newBlock.chunks.push_back(remChunk);
  }

  m_blocks.push_back(std::move(newBlock));

  SuballocResult res;
  res.success = true;
  res.blockIndex = newBlockIdx;
  res.offset = 0;
  res.requiresNewBlock = false;
  return res;
}

bool BlockSuballocator::free(uint32_t blockIndex, uint64_t offset) {
  std::lock_guard<std::mutex> lock(m_mutex);

  if (blockIndex >= m_blocks.size()) {
    return false;
  }

  auto &block = m_blocks[blockIndex];
  bool found = false;
  for (size_t i = 0; i < block.chunks.size(); ++i) {
    if (block.chunks[i].offset == offset && block.chunks[i].inUse) {
      block.chunks[i].inUse = false;
      found = true;
      break;
    }
  }

  if (!found) {
    return false;
  }

  // Coalesce adjacent free chunks
  for (size_t i = 0; i + 1 < block.chunks.size(); ) {
    if (!block.chunks[i].inUse && !block.chunks[i + 1].inUse) {
      block.chunks[i].size += block.chunks[i + 1].size;
      block.chunks.erase(block.chunks.begin() + i + 1);
    } else {
      ++i;
    }
  }

  return true;
}

void BlockSuballocator::clear() {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_blocks.clear();
}

} // namespace gpubench
