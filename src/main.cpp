
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <vector>
#include <array>
#include <optional>
#include <set>
#include <cstdint> 		// Necessary for UINT32_MAX
#include <algorithm> 	// Necessary for std::min/std::max
#include <fstream>

#define ASSERT(conditional, err_msg) if (!conditional) { throw std::runtime_error(err_msg); }
#define VK_ASSERT(conditional, err_msg) if (conditional != VK_SUCCESS) { throw std::runtime_error(err_msg); }

// ----------------------------------------------------------------------------
//  Structs
// ----------------------------------------------------------------------------

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() const
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;			// Basic surface capabilities (min/max number of images in swap chain, min/max width and height of images)
	std::vector<VkSurfaceFormatKHR> formats;		// Surface formats (pixel format, color space)
	std::vector<VkPresentModeKHR> presentModes;		// Available presentation modes
};

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription {};
		bindingDescription.binding = 0;									// Index in the array of bindings ( We're only going to use one )
		bindingDescription.stride = sizeof(Vertex);						// Size of entry
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;		// Move to the next data entry after each vertex ( No-instance rendering )

		return bindingDescription;
	}

	// Describe how to extract a vertex attribute from a chunk of vertex data originating from a binding descriptor.
	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions {};

		// Position
		attributeDescriptions[0].binding = 0;						// Source binding the data is coming from
		attributeDescriptions[0].location = 0;						// Location directive in vertex shader
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;	// 32 bit Vec2
		attributeDescriptions[0].offset = offsetof(Vertex, pos);	// Offset of position attribute

		// Color
		attributeDescriptions[1].binding = 0;							// Source binding the data is coming from
		attributeDescriptions[1].location = 1;							// Location directive in vertex shader
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;	// 32 bit Vec3
		attributeDescriptions[1].offset = offsetof(Vertex, color);		// Offset of color attribute

		return attributeDescriptions;
	}
};


// ----------------------------------------------------------------------------
// Public Functions
// ----------------------------------------------------------------------------

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	// Get function address
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	// Attempt to run
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	// Get function address
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	// Attempt to run
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}

// ----------------------------------------------------------------------------
//  Application
// ----------------------------------------------------------------------------

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:

	// ------------------------------------------------------------------------
	// GLFW API

	GLFWwindow* window = nullptr;
	const uint32_t W_WIDTH = 1080;
	const uint32_t W_HEIGHT = 720;

	void initWindow()
	{
		glfwInit();                                      // Init GLFW lib

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);    // We are not using OpenGL

		// Create window
		window = glfwCreateWindow(W_WIDTH, W_HEIGHT, "Vulkan", nullptr, nullptr);

		// Set user-data pointer for our window
		glfwSetWindowUserPointer(window, this);

		// Resize window callback
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	void cleanupWindow()
	{
		// Destroy native window
		glfwDestroyWindow(window);

		// De-init GLFW library
		glfwTerminate();
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	// ------------------------------------------------------------------------
	// Application API

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
		vkQueueWaitIdle(graphicsQueue);
		vkQueueWaitIdle(presentQueue);
	}

	void cleanup()
	{
		// Cleanup vulkan instance and child resources
		cleanupVulkan();

		// Clean up window and resources
		cleanupWindow();
	}

	static std::vector<char> readFile(const std::string & filename)
	{
		// Open and start reading from the end
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		// Validate
		ASSERT(file.is_open(), "Failed to pen file!");

		// Allocate file buffer
		size_t fileSize = (size_t) file.tellg();
		std::vector<char> buffer(fileSize);

		// Read file into buffer
		file.seekg(0);
		file.read(buffer.data(), fileSize);

		// Close file
		file.close();

		return buffer;
	}

	// ------------------------------------------------------------------------
	// Vulkan API

	const uint32_t MAX_FRAMES_IN_FLIGHT = 2;			// Max frames that can be in flight

	VkInstance instance;								// Connection between application and the VK Library.
	VkDebugUtilsMessengerEXT debugMessenger;			// Debug Callback Messenger Handle
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;	// Physical Device (Graphics card) handle that we will be using
	VkDevice device = VK_NULL_HANDLE;					// Logical Device (Physical Device instance in application) handle that we will be using
	VkQueue graphicsQueue;								// Graphics queues handle (Logical Device)
	VkQueue presentQueue;								// Present queues handle (Logical Device)
	VkSurfaceKHR surface;								// Windows Surface
	VkSwapchainKHR swapChain;							// Window Surface Swap chain
	VkSwapchainKHR oldSwapChain = VK_NULL_HANDLE;		// Retired Window Surface Swap chain
	std::vector<VkImage> swapChainImages;				// Swap chain images
	VkFormat swapChainImageFormat;						// Swap chain image format
	VkExtent2D swapChainExtent;							// Swap chain extents (width and height)
	std::vector<VkImageView> swapChainImageViews;		// Swap chain image views
	VkPipelineLayout pipelineLayout;					// Specified uniform values
	VkRenderPass renderPass;							// Render Pass in Graphics Pipeline
	VkPipeline graphicsPipeline;						// Graphics Pipeline
	std::vector<VkFramebuffer> swapChainFramebuffers;	// Swap Chain Frame-buffers for the images for presentation
	VkCommandPool commandPool;							// Command pool (Used to create command buffers)
	std::vector<VkCommandBuffer> commandBuffers;		// Command buffers for the images in the swap chain
	std::vector<VkSemaphore> imageAvailableSemaphores;	// Semaphore for images (GPU to CPU sync)
	std::vector<VkSemaphore> renderFinishedSemaphores;	// Semaphore for presentation (GPU to CPU sync)
	std::vector<VkFence> inFlightFences;				// Fence for limit max frames (CPU-GPU sync)
	std::vector<VkFence> imagesInFlight;				// Fence for images in flight (CPU-GPU sync)
	size_t currentFrame = 0;							// Current frame index (Used to query semaphores)
	bool framebufferResized = false;					// Flag to determine if window resize has occurred
	float lineWidth = 1.f;								// Primitives line width

	// Logical device required extensions
	const std::vector<const char *> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	// Validation layers are optional components that hook into Vulkan function calls to apply additional operations (Debugging).
	const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

	// Toggle validation layers
	#ifdef NDEBUG
	const bool enableValidationLayers = false;
	#else
	const bool enableValidationLayers = true;
	#endif

	void initVulkan()
	{
		createInstance();

		setupDebugMessenger();

		createSurface();		// Note: This is performed right after the instance creation since it can influence our result when choosing a physical device.

		pickPhysicalDevice();

		createLogicalDevice();

		createSwapChain();

		createImageViews();

		createRenderPass();

		createGraphicsPipeline();

		createFramebuffers();

		createCommandPool();

		createCommandBuffers();
		recordCommandBuffers();

		createSyncObjects();
	}

	void cleanupVulkan()
	{
		cleanupSwapChain();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);	// Destroy semaphores
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);   // Destroy semaphores
			vkDestroyFence(device, inFlightFences[i], nullptr);					// Destroy fences
		}

		vkDestroyCommandPool(device, commandPool, nullptr);				// Destroy command pool

		vkDestroyPipeline(device, graphicsPipeline, nullptr);			// Destroy graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);		// Destroy uniforms

		vkDestroyDevice(device, nullptr);								// Destroy logical device

		// Destroy VK Debug Messenger
		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);				// Destroy window surface
		vkDestroyInstance(instance, nullptr);							// Destroy VK Instance
	}

	// Will perform the following:
	// - Acquire an image from the swap chain
	// - Execute the command buffer with that image as attachment in the frame buffer
	// - Return the image to the swap chain for presentation
	void drawFrame()
	{
		// Wait for previous frame to be finished (And not-in-flight)
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);	// Wait for all the fences

		// - ACQUIRE AN IMAGE FROM THE SWAP CHAIN

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		// Recreate swap chain if out of date (Can't draw out of date swap-chain)
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		else
		{
			ASSERT(result == VK_SUCCESS && result == VK_SUBOPTIMAL_KHR, "Failed to acquire swap chain image!")
		}

		// Check if a previous frame is using this image (i.e. there is its fence to wait on)
		if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
		{
			vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
		}

		// Mark the image as now being in use by this frame
		imagesInFlight[imageIndex] = inFlightFences[currentFrame];

		// - EXECUTE THE COMMAND BUFFER WITH THAT IMAGE AS ATTACHMENT IN THE FRAME BUFFER

		// Setup submission command buffer struct info
		VkSubmitInfo submitInfo {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		// Specify which semaphore to wait on before execution begins and in which pipeline states to wait
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		// Command buffers to actually submit for execution
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		// Which semaphores to signal once the commands buffers have finished execution
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		// Reset fences states
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// Submit graphic command buffer queue
		VK_ASSERT(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]), "Failed to submit draw command buffer!")

		// - RETURN THE IMAGE TO THE SWAP CHAIN FOR PRESENTATION

		// Setup presentation submission struct
		VkPresentInfoKHR presentInfo {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		// Semaphores to wait on before presentation can happen
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		// Swap chains to present images to and the index of the image for each swap chain.
		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		presentInfo.pResults = nullptr;	// Optional

		// Submit the request to present an image to the swap chain
		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		// Recreate swap chain if needed
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else
		{
			ASSERT(result == VK_SUCCESS, "Failed to present swap chain image!")
		}

		// Advance to next frame
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void createSyncObjects()
	{
		// Have enough amount of sync objects
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

		// Setup semaphore creation info
		VkSemaphoreCreateInfo semaphoreInfo {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		// Setup fence creation info
		VkFenceCreateInfo fenceInfo {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		// Create semaphores
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			VK_ASSERT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]), "Failed to create available images semaphore!");
			VK_ASSERT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]), "Failed to create render finished semaphore!");
			VK_ASSERT(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]), "Failed to create fences!");
		}
	}

	void createCommandBuffers()
	{
		{
			// Enough command buffers to cover images in the swap chain
			commandBuffers.resize(swapChainFramebuffers.size());

			// Setup command buffer creation
			VkCommandBufferAllocateInfo allocInfo {};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool = commandPool;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

			// Create command buffers
			VK_ASSERT(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()), "Failed to allocate command buffers!")
		}
	}

	void recordCommandBuffers()
	{
		// Record commands in all the available buffers
		for (size_t i = 0; i < commandBuffers.size(); i++)
		{
			// Begin recording info struct setup
			VkCommandBufferBeginInfo beginInfo {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0;					// Optional
			beginInfo.pInheritanceInfo = nullptr;	// Optional

			// Start recording commands
			VK_ASSERT(vkBeginCommandBuffer(commandBuffers[i], &beginInfo), "Failed to begin recording command buffer!")

			// Setup render pass begin information
			VkRenderPassBeginInfo renderPassInfo {};
			{
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

				// Setup render pass attachments
				renderPassInfo.renderPass = renderPass;
				renderPassInfo.framebuffer = swapChainFramebuffers[i];

				// Render area (Pixel outside this range will have undefined values)
				renderPassInfo.renderArea.offset = { 0, 0 };
				renderPassInfo.renderArea.extent = swapChainExtent;

				// Record clear values
				VkClearValue clearColor = {{{ 0.0f, 0.0f, 0.0f, 1.0f }}};
				renderPassInfo.clearValueCount = 1;
				renderPassInfo.pClearValues = &clearColor;
			}

			// Begin render pass
			// Note: All the functions that record commands have the prefix "vkCmd"
			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// Bind the graphics pipeline
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			// Update frame-buffer viewport
			VkViewport viewport {};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float) swapChainExtent.width;
			viewport.height = (float) swapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

			// Update scissor-test
			VkRect2D scissor{};
			scissor.offset = {0, 0};
			scissor.extent = swapChainExtent;
			vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);;

			// Update line width
			vkCmdSetLineWidth(commandBuffers[i], lineWidth);

			// Main draw call
			vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

			// Render pass is done
			vkCmdEndRenderPass(commandBuffers[i]);

			// Done recording commn
			VK_ASSERT(vkEndCommandBuffer(commandBuffers[i]), "Failed to record command buffers!")
		}
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		// Setup create info for command pool for graphics
		VkCommandPoolCreateInfo poolInfo {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		poolInfo.flags = 0;	// Optional

		// Create command pool
		VK_ASSERT(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "Failed to create command pool!");
	}

	void createFramebuffers()
	{
		// One framebuffer per swap chain image views
		swapChainFramebuffers.resize(swapChainImageViews.size());

		// Create framebuffers
		for (size_t i = 0; i < swapChainImageViews.size(); ++i)
		{
			// Swap chain that will be matched to the framebuffer
			VkImageView attachments[] = { swapChainImageViews[i] };

			// Setup creation info
			VkFramebufferCreateInfo framebufferInfo {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			// Create framebuffer for swap chain image view
			VK_ASSERT(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]), "Failed to crate framebuffer!")
		}
	}

	void createGraphicsPipeline()
	{
		// PROGRAMMABLE PIPELINE (Shader Stages)

		// Load shader bytecode
		auto vertShaderCode = readFile(VK_PRJ_SOURCE_DIR"/assets/shaders/vert.spv");
		auto fragShaderCode = readFile(VK_PRJ_SOURCE_DIR"/assets/shaders/frag.spv");

		// Create shader modules
		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// Vertex shader stage setup
		VkPipelineShaderStageCreateInfo vertShaderStageInfo {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;								// Shader stage 
		vertShaderStageInfo.module = vertShaderModule;										// Shader module contained the code
		vertShaderStageInfo.pName = "main";													// Shader entrypoint
		vertShaderStageInfo.pSpecializationInfo = nullptr;									// No constants

		// Fragment shader stage setup
		VkPipelineShaderStageCreateInfo fragShaderStageInfo {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;							// Shader stage
		fragShaderStageInfo.module = fragShaderModule;										// Shader module contained the code
		fragShaderStageInfo.pName = "main";													// Shader entrypoint
		fragShaderStageInfo.pSpecializationInfo = nullptr;									// No constants

		// Shader stages for graphics pipeline (Programmable Pipeline Stages)
		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		// FIXED-PIPELINE

		// Struct that describes the format of the vertex data that will be passed to the vertex shader
		VkPipelineVertexInputStateCreateInfo vertexInputInfo {};
		{
			auto bindingDescription = Vertex::getBindingDescription();
			auto attributeDescriptions = Vertex::getAttributeDescriptions();

			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount = 1;
			vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;

			vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
		}

		// Struct describes two things: what kind of geometry will be drawn from the vertices and if primitive restart should be enabled.
		VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		// Frame-buffer viewport
		VkViewport viewport {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float) swapChainExtent.width;
		viewport.height = (float) swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		// Enable scissor-test
		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swapChainExtent;

		// Setup viewport given scissor and viewport
		VkPipelineViewportStateCreateInfo viewportState {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		// Setup fragment rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; 	// Optional
		rasterizer.depthBiasClamp = 0.0f; 			// Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; 	// Optional

		// Multi-sampling
		VkPipelineMultisampleStateCreateInfo multisampling {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; 						// Optional
		multisampling.pSampleMask = nullptr; 						// Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; 			// Optional
		multisampling.alphaToOneEnable = VK_FALSE; 					// Optional

		// Struct that contains the configuration per attached framebuffer.
		VkPipelineColorBlendAttachmentState colorBlendAttachment {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; 	// Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; 	// Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; 				// Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; 	// Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; 	// Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; 				// Optional

		// Struct that contains the global color blending settings
		VkPipelineColorBlendStateCreateInfo colorBlending {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; 			// Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; 			// Optional
		colorBlending.blendConstants[1] = 0.0f; 			// Optional
		colorBlending.blendConstants[2] = 0.0f; 			// Optional
		colorBlending.blendConstants[3] = 0.0f; 			// Optional

		// Specify dynamic states of the pipeline
		VkDynamicState dynamicStates[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
				VK_DYNAMIC_STATE_LINE_WIDTH
		};

		VkPipelineDynamicStateCreateInfo dynamicState {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = 3;
		dynamicState.pDynamicStates = dynamicStates;

		// Setup uniform values struct
		VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0; 				// Optional
		pipelineLayoutInfo.pSetLayouts = nullptr; 			// Optional
		pipelineLayoutInfo.pushConstantRangeCount = 0; 		// Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; 	// Optional

		// Create uniforms layout
		VK_ASSERT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "Failed to create pipeline layout!");

		// Setup graphics pipeline info struct
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		{
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

			// Shader stages
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = shaderStages;

			// Fixed pipeline
			pipelineInfo.pVertexInputState = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssembly;
			pipelineInfo.pViewportState = &viewportState;
			pipelineInfo.pRasterizationState = &rasterizer;
			pipelineInfo.pMultisampleState = &multisampling;
			pipelineInfo.pDepthStencilState = nullptr;			// Optional
			pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.pDynamicState = &dynamicState;        	// Optional

			// Uniforms
			pipelineInfo.layout = pipelineLayout;

			// Render-pass
			pipelineInfo.renderPass = renderPass;
			pipelineInfo.subpass = 0;                			// Note: Index

			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;   // Optional
			pipelineInfo.basePipelineIndex = -1;                // Optional
		}

		VK_ASSERT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline), "Failed to create graphics pipeline!")

		// Clean up shader modules since Graphics Pipeline has been created.
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createRenderPass()
	{
		// One single color buffer attachment
		VkAttachmentDescription colorAttachment {};
		colorAttachment.format = swapChainImageFormat;						// Color attachment format should match the format of the images in the swap chain
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;					// No multisampling (So far)
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;				// What to do with data before rendering
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;				// What to do with data after rendering
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;	// Don't care what happens to data
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;	// Don't care what happens to data
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;			// Layout the image will have before the render pass begins
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;		// Layout to automatically transition to when the render pass finishes

		// Attachment reference for the sub-pass
		VkAttachmentReference colorAttachmentRef {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// One sub-pass in the render pass
		VkSubpassDescription subpass {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		// Sub-pass dependency
		VkSubpassDependency dependency {};
		{
			// Specify the indices of the dependency and the dependent subpass
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;

			// Specify the operations to wait on and the stages in which these operations occur
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;

			// The operations that should wait on this are in the color attachment stage and involve the writing of the color attachment
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		}

		// Render pass setup
		VkRenderPassCreateInfo renderPassInfo {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		VK_ASSERT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), "Failed to create render passes!");
	}

	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		VK_ASSERT(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule), "Failed to create shader module!")

		return shaderModule;
	}

	void createImageViews()
	{
		// Ensure proper size of image view container
		swapChainImageViews.resize(swapChainImages.size());

		// Create all the image views needed
		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			VkImageViewCreateInfo createInfo {};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];

			// Format for a 2D texture
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;

			// Default color channel mapping
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			// Our images will be used as color targets without any mipmapping levels or multiple layers
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			// Create view for the given swap chain image
			VK_ASSERT(vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]), "Failed to create image views!");
		}
	}

	void createSwapChain()
	{
		// Get supported swap chain information
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		// Setup basic swap chain properties
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		// Minimum number of images required for swap chain to function
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		// Ensure we don't exceed the max number of images supported (0 signifies that there is no maximum)
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// Swap chain creation info setup
		VkSwapchainCreateInfoKHR createInfo {};
		{
			createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			createInfo.surface = surface;
			createInfo.minImageCount = imageCount;
			createInfo.imageFormat = surfaceFormat.format;
			createInfo.imageColorSpace = surfaceFormat.colorSpace;
			createInfo.imageExtent = extent;
			createInfo.imageArrayLayers = 1;                                // Note: Always one unless developing a stereoscopic 3D application
			createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			// Setup how to handle swap chain images that will be used across multiple queue families.
			QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
			uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

			if (indices.graphicsFamily != indices.presentFamily)
			{
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
			}
			else
			{
				createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
				createInfo.queueFamilyIndexCount = 0;        // Optional
				createInfo.pQueueFamilyIndices = nullptr;    // Optional
			}

			createInfo.preTransform = swapChainSupport.capabilities.currentTransform;    // No transformation performed
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			createInfo.presentMode = presentMode;
			createInfo.clipped = VK_TRUE;
			createInfo.oldSwapchain = oldSwapChain;
		}

		// Create swap chain
		VK_ASSERT(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain), "Failed to create swap chain!")

		// Destroy old swap chain
		if (oldSwapChain != VK_NULL_HANDLE)
		{
			vkDestroySwapchainKHR(device, oldSwapChain, nullptr);
			oldSwapChain = VK_NULL_HANDLE;
		}

		// Get available images in the swap chain (Since we only specify a minimum, the implementation is more than welcome to create more than enough)
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		// Store other relevant swap chain information
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
		oldSwapChain = swapChain;
	}

	void recreateSwapChain()
	{
		// Pause during window minimization
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);

		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		// Wait for resources in use
		vkDeviceWaitIdle(device);

		// Create swap chain based on old swap chain and clean up old swap chain dependent resources
		createSwapChain();
		cleanupSwapChainDependentResources();

		// Create new swap chain dependent resources
		createImageViews();
		createRenderPass();
		createFramebuffers();
		createCommandBuffers();
		recordCommandBuffers();

		// Have enought amount of fences for swap chain images
		imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
	}

	void cleanupSwapChainDependentResources()
	{
		// Destroy frame buffers (Before image views and render pass they are based on)
		for (auto framebuffer : swapChainFramebuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		// Destroy command buffers (No need to destroy command bool)
		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

		vkDestroyRenderPass(device, renderPass, nullptr);				// Destroy render pass

		// Destroy swap chain image views
		for (auto imageView : swapChainImageViews)
		{
			vkDestroyImageView(device, imageView, nullptr);
		}
	}

	void cleanupSwapChain()
	{
		// Clean up resources
		cleanupSwapChainDependentResources();

		// Destroy current window swap-chain
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	// Surface format is the color depth of our surface.
	// Note: For the color space we'll use SRGB if it is available, because it results in more accurate perceived colors.
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> & availableFormats)
	{
		// Check for desired format combination availability
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}

		// We could rank the "best" available formats but not needed for this project.

		return availableFormats[0];
	}

	// The presentation mode is arguably the most important setting for the swap chain, because it represents the actual conditions for showing images to the screen
	// Modes: VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR, VK_PRESENT_MODE_MAILBOX_KHR.
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> & availablePresentModes)
	{
		// Favor the Mailbox present mode (if available)
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		// Default to the guaranteed supported present mode
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	// The swap extent is the resolution of the swap chain images and it's almost always exactly equal to the resolution of the window that we're drawing to in pixels
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	void createSurface()
	{
		VK_ASSERT(glfwCreateWindowSurface(instance, window, nullptr, &surface), "Failed to create window surface!");
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		// Create queues for all supported families
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;

			queueCreateInfos.push_back(queueCreateInfo);
		}

		// Query for the physical device features that we will be using
		VkPhysicalDeviceFeatures deviceFeatures {};
		deviceFeatures.wideLines = true;

		// Create logical device
		VkDeviceCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		// Support older implementations where distinction between instance and device validation layers exist

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		// Create logical device
		VK_ASSERT(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device), "Failed to create logical device!")

		// Query queue handles from the device
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void pickPhysicalDevice()
	{
		// Get GPU count
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		ASSERT((deviceCount > 0), "Failed to find GPUs that support Vulkan!")

		// Get physical devices (GPUs)
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		// Check if physical devices meet the requirements we have
		for (const auto& device: devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}
	}

	bool isDeviceSuitable(VkPhysicalDevice device)
	{
//		// Query physical device its info
//		VkPhysicalDeviceProperties deviceProperties;
//		vkGetPhysicalDeviceProperties(device, &deviceProperties);
//
//		// Query physical device for optional features
//		VkPhysicalDeviceFeatures deviceFeatures;
//		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		// Check for logical device for extension support
		bool extensionsSupported = checkDeviceExtensionsSupport(device);

		// Check for swap-chain compatibility
		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		// Physical device has to have at least a graphics queue family to be supported
		QueueFamilyIndices indices = findQueueFamilies(device);

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		// Query device for basic surface capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		// Get surface format count
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		// Query surface formats (if any are available)
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		// Get present mode count
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		// Query device present modes (if any are available)
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	bool checkDeviceExtensionsSupport(VkPhysicalDevice device)
	{
		// Get supported extension count
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		// Get name list of supported extensions
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		// Check that all the required extensions are part of the available extensions
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto & extension : availableExtensions)
			requiredExtensions.erase(extension.extensionName);


		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		// Get queue family count
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		// Get list of queue families
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// We need to find at least one queue family that support VK_QUEUE_GRAPHICS_BIT
		for (uint32_t i = 0; i < queueFamilyCount; ++i)
		{
			// Check for graphics queue support
			if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}

			// Check for present queue support
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport)
				indices.presentFamily = i;

			// Queue families where found
			if (indices.isComplete())
				break;
		}

		return indices;
	}

	void createInstance()
	{
		// Enable validation layers
		ASSERT(enableValidationLayers && !checkValidationLayerSupport(), "Validation layers requested, but not available!")

		// Setup application info
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// Get required extensions
		auto extensions = getRequiredExtensions();

		// Setup creation info
		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			// Setup layers
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			// Setup message debugger to vkCreateInstance() and vkDestroyInstance()
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		// Create VK Instance
		VK_ASSERT(vkCreateInstance(&createInfo, nullptr, &instance), "Failed to create instance!");

		// Show available and supported extensions
		displayExtensionsInfo();
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = this;
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers) return;

		// Setup debug messenger creation info struct
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);

		// Create debug messenger
		VK_ASSERT(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger), "Failed to set up debug messenger!");
	}

	// Validation layer debug callback
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
														VkDebugUtilsMessageTypeFlagsEXT messageType,
														const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
														void* pUserData)
	{
		if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
		{
			// Message is important enough to show
			std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
		}

		return VK_FALSE;
	}

	bool checkValidationLayerSupport()
	{
		// Get supported layer count
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		// Get supported layers
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		// Ensure layers are supported
		for (const char* layerName: validationLayers)
		{
			bool layerFound = false;

			// Attempt to validate layer
			for (const auto& layerProperties: availableLayers)
			{
				if (std::strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			// Layer is not supported
			if (!layerFound)
			{
				return false;
			}
		}

		// All the validation layers are supported
		return true;
	}

	/// Get required list of extensions based on whether validation layers are enabled or not:
	std::vector<const char*> getRequiredExtensions()
	{
		// Get GLFW required extensions from Vulkan to handle the native window
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		// Debug Utilities extension
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		// Validation layer required extension
		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	void displayExtensionsInfo()
	{
		// Get GLFW required extensions from Vulkan to handle the native window
		auto requiredExtensions = getRequiredExtensions();

		// Check for supported VK Extensions
		uint32_t extensionCount = 0;
		VK_ASSERT(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "Failed to get supported extensions count");

		std::vector<VkExtensionProperties> extensions(extensionCount);
		VK_ASSERT(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()), "Failed to get supported extensions properties");

		// Log required GLFW Extensions
		std::cout << "Required extensions:\n";
		for (auto& requiredExtension: requiredExtensions)
		{
			uint8_t isSupported = 0;

			// Check GLFW Extension is supported
			for (const auto& extension: extensions)
			{
				if (std::strcmp(extension.extensionName, requiredExtension) == 0)
				{
					isSupported = 1;
					break;
				}
			}

			std::cout << '\t' << "- " << requiredExtension << " | " << (isSupported ? "SUPPORTED" : "NOT SUPPORTED") << '\n';
		}

		// Log Device's available extensions
		std::cout << "\nAvailable extensions:\n";
		for (const auto& extension: extensions)
		{
			std::cout << '\t' << "- " << extension.extensionName << '\n';
		}
	}
};

// ----------------------------------------------------------------------------
//  Application entry point
// ----------------------------------------------------------------------------

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
