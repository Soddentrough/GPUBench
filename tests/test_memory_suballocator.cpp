#include "test_harness.h"
#include "core/BlockSuballocator.h"

using namespace gpubench;

TEST_CASE(BlockSuballocator, InitialStateEmpty) {
  BlockSuballocator alloc;
  ASSERT_EQ(alloc.getBlocks().size(), 0u);

  // Attempting to allocate from empty suballocator signals requiresNewBlock
  auto res = alloc.allocate(1024 * 1024, 256, 0, false);
  ASSERT_FALSE(res.success);
  ASSERT_TRUE(res.requiresNewBlock);
}

TEST_CASE(BlockSuballocator, SingleBlockAllocationAndAlignment) {
  BlockSuballocator alloc;
  constexpr uint64_t kBlockSize = 64ULL * 1024ULL * 1024ULL; // 64 MB

  // Register block 0 with 4 MB allocation
  auto res1 = alloc.registerNewBlock(4 * 1024 * 1024, kBlockSize, 0, false);
  ASSERT_TRUE(res1.success);
  ASSERT_EQ(res1.blockIndex, 0u);
  ASSERT_EQ(res1.offset, 0u);

  const auto &blocks = alloc.getBlocks();
  ASSERT_EQ(blocks.size(), 1u);
  // Expect 2 chunks: [0..4MB, inUse=true], [4MB..64MB, inUse=false]
  ASSERT_EQ(blocks[0].chunks.size(), 2u);
  ASSERT_EQ(blocks[0].chunks[0].size, 4ULL * 1024ULL * 1024ULL);
  ASSERT_TRUE(blocks[0].chunks[0].inUse);
  ASSERT_EQ(blocks[0].chunks[1].size, 60ULL * 1024ULL * 1024ULL);
  ASSERT_FALSE(blocks[0].chunks[1].inUse);

  // Second allocation: 8 MB with 2 MB alignment
  constexpr uint64_t k2MB = 2ULL * 1024ULL * 1024ULL;
  auto res2 = alloc.allocate(8 * 1024 * 1024, k2MB, 0, false);
  ASSERT_TRUE(res2.success);
  ASSERT_EQ(res2.blockIndex, 0u);
  // 4 MB is already a multiple of 2 MB, so alignedStart = 4 MB
  ASSERT_EQ(res2.offset, 4ULL * 1024ULL * 1024ULL);
  ASSERT_EQ(res2.offset % k2MB, 0u);
}

TEST_CASE(BlockSuballocator, AlignmentPaddingCreation) {
  BlockSuballocator alloc;
  constexpr uint64_t kBlockSize = 64ULL * 1024ULL * 1024ULL;

  // Allocate 1 MB + 128 bytes
  uint64_t firstSize = 1024 * 1024 + 128;
  alloc.registerNewBlock(firstSize, kBlockSize, 0, false);

  // Allocate 2 MB with 64 KB alignment
  uint64_t align64K = 64 * 1024;
  auto res = alloc.allocate(2 * 1024 * 1024, align64K, 0, false);
  ASSERT_TRUE(res.success);
  ASSERT_EQ(res.offset % align64K, 0u);
  ASSERT_GT(res.offset, firstSize);

  // Check that padding chunk was retained
  const auto &chunks = alloc.getBlocks()[0].chunks;
  bool foundPadding = false;
  for (const auto &c : chunks) {
    if (!c.inUse && c.offset == firstSize && c.offset + c.size == res.offset) {
      foundPadding = true;
      break;
    }
  }
  ASSERT_TRUE(foundPadding);
}

TEST_CASE(BlockSuballocator, FreeAndCoalesce) {
  BlockSuballocator alloc;
  constexpr uint64_t kBlockSize = 64ULL * 1024ULL * 1024ULL;

  // Allocate 3 contiguous 4 MB chunks: A, B, C
  auto resA = alloc.registerNewBlock(4 * 1024 * 1024, kBlockSize, 0, false);
  auto resB = alloc.allocate(4 * 1024 * 1024, 256, 0, false);
  auto resC = alloc.allocate(4 * 1024 * 1024, 256, 0, false);

  ASSERT_TRUE(resA.success);
  ASSERT_TRUE(resB.success);
  ASSERT_TRUE(resC.success);

  // Free middle chunk B
  ASSERT_TRUE(alloc.free(0, resB.offset));
  {
    const auto &chunks = alloc.getBlocks()[0].chunks;
    // Chunks should be: A(inUse), B(free), C(inUse), Remainder(free)
    ASSERT_EQ(chunks.size(), 4u);
    ASSERT_TRUE(chunks[0].inUse);
    ASSERT_FALSE(chunks[1].inUse);
    ASSERT_TRUE(chunks[2].inUse);
    ASSERT_FALSE(chunks[3].inUse);
  }

  // Free leading chunk A -> should coalesce with B into one free chunk of 8 MB
  ASSERT_TRUE(alloc.free(0, resA.offset));
  {
    const auto &chunks = alloc.getBlocks()[0].chunks;
    // Chunks: [A+B](free, 8MB), C(inUse, 4MB), Remainder(free, 52MB)
    ASSERT_EQ(chunks.size(), 3u);
    ASSERT_FALSE(chunks[0].inUse);
    ASSERT_EQ(chunks[0].size, 8ULL * 1024ULL * 1024ULL);
    ASSERT_TRUE(chunks[1].inUse);
    ASSERT_FALSE(chunks[2].inUse);
  }

  // Free chunk C -> all chunks should coalesce into 1 single 64 MB chunk!
  ASSERT_TRUE(alloc.free(0, resC.offset));
  {
    const auto &chunks = alloc.getBlocks()[0].chunks;
    ASSERT_EQ(chunks.size(), 1u);
    ASSERT_FALSE(chunks[0].inUse);
    ASSERT_EQ(chunks[0].offset, 0u);
    ASSERT_EQ(chunks[0].size, kBlockSize);
  }
}

TEST_CASE(BlockSuballocator, ExhaustionTriggersNewBlock) {
  BlockSuballocator alloc;
  constexpr uint64_t kBlockSize = 64ULL * 1024ULL * 1024ULL;

  // Allocate 60 MB
  auto res1 = alloc.registerNewBlock(60 * 1024 * 1024, kBlockSize, 0, false);
  ASSERT_TRUE(res1.success);

  // Try to allocate 10 MB in block with only 4 MB left
  auto res2 = alloc.allocate(10 * 1024 * 1024, 256, 0, false);
  ASSERT_FALSE(res2.success);
  ASSERT_TRUE(res2.requiresNewBlock);

  // Register block 1 and allocate 10 MB
  auto res3 = alloc.registerNewBlock(10 * 1024 * 1024, kBlockSize, 0, false);
  ASSERT_TRUE(res3.success);
  ASSERT_EQ(res3.blockIndex, 1u);
  ASSERT_EQ(alloc.getBlocks().size(), 2u);
}
