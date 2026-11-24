#pragma once

#include "core/VulkanContext.h"
#include "showcase/ShowcaseMath.h"
#include "showcase/ShowcaseScene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpubench::showcase {

enum class RenderMode : uint32_t {
  Naive = 0u,
  Legacy = 1u,
  Modern = 2u,
};

enum class OverlayMode : uint32_t {
  Beauty = 0u,
  Overdraw = 1u,
  MicroTriangles = 2u,
  MeshletClusters = 3u,
};

struct OptimizationFlags {
  bool adaptiveLod{true};
  bool frustumConeCull{true};
  bool hizOcclusion{true};
  bool visBuffer{true};
  bool dgcMaterialBinning{true};

  uint32_t toBitmask() const {
    uint32_t mask = 0u;
    if (adaptiveLod) mask |= (1u << 0);
    if (frustumConeCull) mask |= (1u << 1);
    if (hizOcclusion) mask |= (1u << 2);
    if (visBuffer) mask |= (1u << 3);
    if (dgcMaterialBinning) mask |= (1u << 4);
    return mask;
  }
};

struct ShowcaseSettings {
  RenderMode mode{RenderMode::Modern};
  OverlayMode overlay{OverlayMode::Beauty};
  OptimizationFlags opts{};
  bool freezeCulling{false};
  bool splitWipeCompare{false};
  RenderMode splitLeftMode{RenderMode::Naive};
  float splitWipeX{0.5f};
  float tessLevelMax{64.0f};
  CameraState activeCam{};
  CameraState frozenCam{};
};

struct FrameTelemetry {
  double totalMs{0.0};
  double geometryMs{0.0};
  double cullHizMs{0.0};
  double shadingMs{0.0};
  double fps{0.0};
  double smoothedMs{0.0};
  double smoothedFps{0.0};

  static constexpr uint32_t kHistoryLen = 120u;
  float fpsHistory[kHistoryLen]{};
  float msHistory[kHistoryLen]{};
  uint32_t historyCursor{0};
  uint32_t historyCount{0};

  uint32_t drawCalls{0};
  uint32_t totalMeshlets{0};
  uint32_t renderedMeshlets{0};
  uint32_t lodCulled{0};
  uint32_t frustumCulled{0};
  uint32_t coneCulled{0};
  uint32_t hizCulled{0};
  double culledPct{0.0};
  uint32_t renderedTriangles{0};
  uint32_t activeDgcBins{0};

  double avgOverdraw{1.0};
  double subPixelTriPct{0.0};
  uint64_t clippedPrimitives{0};
  uint64_t fragmentInvocations{0};

  bool hasParitySample{false};
  double parityPsnr{100.0};
  double parityExactPct{100.0};
  double parityNearPct{100.0};

  bool meshShaderSupported{false};
  bool dgcExecutionSetSupported{false};
  bool barycentricSupported{false};
  bool tessellationSupported{false};

  bool meshShaderActive{false};
  bool dgcExecutionSetActive{false};
  bool barycentricActive{false};
  bool tessellationActive{false};
};

struct ThreeWayResult {
  double naiveMs{0.0}, naiveFps{0.0};
  uint32_t naiveDrawCalls{0};
  uint64_t naivePrimitives{0};
  double naiveOverdraw{1.0}, naiveSubPxPct{0.0};

  double legacyMs{0.0}, legacyFps{0.0};
  uint32_t legacyDrawCalls{0};
  uint64_t legacyPrimitives{0};
  double legacyOverdraw{1.0}, legacySubPxPct{0.0};

  double modernMs{0.0}, modernFps{0.0};
  uint32_t modernDrawCalls{0};
  uint64_t modernPrimitives{0};
  double modernCulledPct{0.0}, modernOverdraw{1.0}, modernSubPxPct{0.0};

  double modernVsNaiveSpeedup{1.0};
  double modernVsLegacySpeedup{1.0};
  double parityPsnr{100.0};
  double parityExactPct{100.0};
  bool valid{false};
};

class ShowcaseRenderer {
public:
  ShowcaseRenderer() = default;
  ~ShowcaseRenderer();

  ShowcaseRenderer(const ShowcaseRenderer &) = delete;
  ShowcaseRenderer &operator=(const ShowcaseRenderer &) = delete;

  bool init(::VulkanContext *ctx, const std::string &kernelDir, uint32_t initialW = 1280u, uint32_t initialH = 720u);
  void shutdown();
  bool isInitialized() const { return m_initialized; }

  void resize(uint32_t width, uint32_t height);
  void updateSettings(const ShowcaseSettings &settings) { m_settings = settings; }
  const ShowcaseSettings &getSettings() const { return m_settings; }
  const ShowcaseSceneData &getScene() const { return m_scene; }

  // Records a live viewport frame inside the active swapchain command buffer (ends with displayColorImage in SHADER_READ_ONLY_OPTIMAL)
  void recordFrame(VkCommandBuffer cmd, uint32_t frameSlot);

  // Synchronous offscreen single-mode render for automated verification & 3-way benchmarks
  bool renderHeadlessFrame(const ShowcaseSettings &settings, uint32_t width, uint32_t height,
                           FrameTelemetry &outTelemetry, std::vector<uint8_t> *outRgba = nullptr);

  ThreeWayResult runThreeWayBenchmark(const CameraState &cam, uint32_t width, uint32_t height);

  VkImageView getDisplayImageView() const { return m_displayColor.view; }
  uint32_t getDisplayWidth() const { return m_width; }
  uint32_t getDisplayHeight() const { return m_height; }
  uint64_t getImageGeneration() const { return m_imageGeneration; }
  const FrameTelemetry &getTelemetry() const { return m_telemetry; }

private:
  static constexpr uint32_t kMaxSlots = 6u;
  static constexpr uint32_t kShadowSize = 2048u;
  static constexpr uint32_t kMaxFallbackDraws = 131072u;

  struct GpuBuffer {
    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    void *mapped{nullptr};
    VkDeviceSize size{0};
    VkDeviceAddress address{0};
  };

  struct GpuImage {
    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    std::vector<VkImageView> mipViews;
    VkFormat format{VK_FORMAT_UNDEFINED};
    uint32_t width{0};
    uint32_t height{0};
    uint32_t mipLevels{1};
    VkImageLayout currentLayout{VK_IMAGE_LAYOUT_UNDEFINED};
  };

  struct GpuUniforms {
    float viewProj[16];
    float prevViewProj[16];
    float cullViewProj[16];
    float shadowViewProj[16];
    float cullFrustumPlanes[6][4];
    float camPosFov[4];
    float cullCamPos[4];
    float sunDirIntensity[4];
    uint32_t screenInfo[4];
    uint32_t countsInfo[4];
    float miscParams[4];
    uint32_t frameMeta[4];
  };

  struct GpuStatsReadback {
    uint32_t totalMeshlets;
    uint32_t renderedMeshlets;
    uint32_t lodCulled;
    uint32_t frustumCulled;
    uint32_t coneCulled;
    uint32_t hizCulled;
    uint32_t renderedTriangles;
    uint32_t activeDgcBins;
    uint32_t overdrawSum;
    uint32_t subPixelTriCount;
    uint32_t totalShadedPixels;
    uint32_t totalFragments;
    uint32_t paritySqErrQuant;
    uint32_t parityExactPixels;
    uint32_t parityNearPixels;
    uint32_t parityTotalPixels;
  };

  struct FrameSlot {
    GpuBuffer uboBuf;
    GpuBuffer statsBuf;
    GpuBuffer legacyIndirectBuf;
    VkDescriptorSet frameSet{VK_NULL_HANDLE};
    VkQueryPool tsPool{VK_NULL_HANDLE};
    VkQueryPool pipeStatsPool{VK_NULL_HANDLE};
    bool submitted{false};
    bool usedPipeStats{false};
    RenderMode recordedMode{RenderMode::Modern};
    OptimizationFlags recordedOpts{};
    uint32_t recordedDrawCalls{0};
    bool splitWipeActive{false};
  };

  bool createSceneGpuResources();
  bool createPipelines(const std::string &kernelDir);
  bool createDgcExecutionSet();
  bool createRenderTargets(uint32_t w, uint32_t h);
  void destroyRenderTargets();
  bool createFrameSlots();
  void destroyFrameSlots();
  void renderShadowMapOnce();

  GpuBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible);
  void destroyBuffer(GpuBuffer &buf);
  void uploadToDeviceBuffer(GpuBuffer &dst, const void *data, VkDeviceSize size);

  GpuImage createImage(uint32_t w, uint32_t h, uint32_t mipLevels, VkFormat fmt, VkImageUsageFlags usage,
                       VkImageAspectFlags aspect);
  void destroyImage(GpuImage &img);
  VkShaderModule loadShaderModule(const std::string &path);

  void transitionImage(VkCommandBuffer cmd, GpuImage &img, VkImageLayout newLayout,
                       VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                       VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                       VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

  void recordModePass(VkCommandBuffer cmd, uint32_t slotIdx, RenderMode mode, const ShowcaseSettings &cfg,
                      bool writeCpuUbo, bool isMeasuredPass);
  void readbackCompletedSlot(uint32_t slotIdx, FrameTelemetry &outTel, bool updateHistory);

  ::VulkanContext *m_ctx{nullptr};
  VkDevice m_device{VK_NULL_HANDLE};
  VkPhysicalDevice m_physDev{VK_NULL_HANDLE};
  VmaAllocator m_vma{VK_NULL_HANDLE};
  bool m_initialized{false};
  uint32_t m_width{0};
  uint32_t m_height{0};
  uint64_t m_imageGeneration{1};
  uint32_t m_frameCounter{0};

  ShowcaseSceneData m_scene;
  ShowcaseSettings m_settings{};
  FrameTelemetry m_telemetry{};
  Mat4 m_prevViewProj{Mat4::identity()};
  Mat4 m_shadowViewProj{Mat4::identity()};

  // Static Scene Buffers (Set 0)
  GpuBuffer m_vb, m_ib, m_meshletBuf, m_meshletVertBuf, m_meshletTriBuf;
  GpuBuffer m_instanceBuf, m_materialBuf, m_lightBuf, m_taskPkt0Buf, m_taskPkt1Buf;
  std::vector<GpuImage> m_sceneTextures;
  VkSampler m_texSampler{VK_NULL_HANDLE};
  VkSampler m_shadowSampler{VK_NULL_HANDLE};
  VkSampler m_hizSampler{VK_NULL_HANDLE};
  VkSampler m_pointSampler{VK_NULL_HANDLE};

  // Shared Dynamic Frame Buffers & Images (Set 1 / Set 2)
  GpuBuffer m_worklistBuf, m_dgcBuf, m_fallbackBuf;
  GpuImage m_shadowMap, m_depthImg, m_hizImg;
  GpuImage m_gbuf0, m_gbuf1, m_gbuf2;
  GpuImage m_visBufImg, m_overdrawImg, m_triAreaImg, m_debugTagImg;
  GpuImage m_sceneColorImg, m_altColorImg, m_displayColor;

  // Descriptors
  VkDescriptorPool m_descPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_sceneSetLayout{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_frameSetLayout{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_hizSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet m_sceneSet{VK_NULL_HANDLE};
  std::vector<VkDescriptorSet> m_hizMipSets;

  // Pipeline Layouts & Pipelines
  VkPipelineLayout m_gfxLayout{VK_NULL_HANDLE};
  VkPipelineLayout m_taskLayout{VK_NULL_HANDLE};
  VkPipelineLayout m_compLayout{VK_NULL_HANDLE};
  VkPipelineLayout m_hizLayout{VK_NULL_HANDLE};
  VkPipelineLayout m_visShadeSpecLayout{VK_NULL_HANDLE};

  VkPipeline m_pipeShadow{VK_NULL_HANDLE};
  VkPipeline m_pipeSky{VK_NULL_HANDLE};
  VkPipeline m_pipeNaiveFlat{VK_NULL_HANDLE};
  VkPipeline m_pipeNaiveTess{VK_NULL_HANDLE};
  VkPipeline m_pipeLegacyFlat{VK_NULL_HANDLE};
  VkPipeline m_pipeLegacyTess{VK_NULL_HANDLE};
  VkPipeline m_pipeLegacyLight{VK_NULL_HANDLE};
  VkPipeline m_pipeModernVisOpaque{VK_NULL_HANDLE};
  VkPipeline m_pipeModernVis{VK_NULL_HANDLE};
  VkPipeline m_pipeModernFwd{VK_NULL_HANDLE};
  VkPipeline m_pipeFallbackCull{VK_NULL_HANDLE};
  VkPipeline m_pipeFallbackVis{VK_NULL_HANDLE};
  VkPipeline m_pipeHizReduce{VK_NULL_HANDLE};
  VkPipeline m_pipeVisClassify{VK_NULL_HANDLE};
  VkPipeline m_pipeVisResolve{VK_NULL_HANDLE};
  VkPipeline m_pipeVisShadeUnified{VK_NULL_HANDLE};
  std::array<VkPipeline, 8> m_pipeVisShadeSpec{};
  VkPipeline m_pipeOverlayCompose{VK_NULL_HANDLE};
  VkPipeline m_pipeFrustumWire{VK_NULL_HANDLE};

  // DGC Execution Set for 8 specialized Material Archetype shaders
  bool m_dgcExecSetReady{false};
  VkIndirectCommandsLayoutEXT m_dgcLayout{VK_NULL_HANDLE};
  VkIndirectExecutionSetEXT m_dgcExecSet{VK_NULL_HANDLE};
  GpuBuffer m_dgcPreprocessBuf;
  VkDeviceMemory m_dgcPreprocessMem{VK_NULL_HANDLE};

  std::array<FrameSlot, kMaxSlots> m_slots{};

  // Transient upload/headless command pool
  VkCommandPool m_uploadCmdPool{VK_NULL_HANDLE};
  VkCommandBuffer m_headlessCmd{VK_NULL_HANDLE};
  VkFence m_headlessFence{VK_NULL_HANDLE};
};

} // namespace gpubench::showcase
