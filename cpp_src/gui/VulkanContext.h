#pragma once

#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

#include <string>
#include <vector>
#include <cstdint>

namespace gpubench::gui {

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    bool init(const char* title, int width, int height, float scaleOverride = 0.0f);
    void shutdown();

    float getDisplayScale() const { return m_displayScale; }

    void beginFrame();
    void endFrame();
    void resize(int width, int height);

    SDL_Window* getWindow() const { return m_window; }
    VkInstance getInstance() const { return m_instance; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VkDevice getDevice() const { return m_device; }
    uint32_t getQueueFamily() const { return m_queueFamily; }
    VkQueue getQueue() const { return m_queue; }
    VkDescriptorPool getDescriptorPool() const { return m_descriptorPool; }

    bool isInitialized() const { return m_initialized; }

private:
    bool setupVulkan();
    void setupVulkanWindow(int width, int height);
    void cleanupVulkanWindow();
    void frameRender(ImDrawData* drawData);
    void framePresent();

    SDL_Window* m_window{nullptr};
    VkAllocationCallbacks* m_allocator{nullptr};
    VkInstance m_instance{VK_NULL_HANDLE};
    VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
    VkDevice m_device{VK_NULL_HANDLE};
    uint32_t m_queueFamily{static_cast<uint32_t>(-1)};
    VkQueue m_queue{VK_NULL_HANDLE};
    VkPipelineCache m_pipelineCache{VK_NULL_HANDLE};
    VkDescriptorPool m_descriptorPool{VK_NULL_HANDLE};

    ImGui_ImplVulkanH_Window m_mainWindowData;
    uint32_t m_minImageCount{2};
    bool m_swapchainRebuild{false};
    bool m_initialized{false};
    float m_displayScale{1.0f};
    VkClearColorValue m_clearColor{{0.08f, 0.09f, 0.12f, 1.00f}};
};

} // namespace gpubench::gui
