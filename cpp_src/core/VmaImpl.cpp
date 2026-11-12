#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>

#ifdef VK_VERSION_1_4
#define VMA_VULKAN_VERSION 1004000
#else
#define VMA_VULKAN_VERSION 1003000
#endif

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#endif
