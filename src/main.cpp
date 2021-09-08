
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <vector>
#include <optional>

#define ASSERT(conditional, err_msg) if (!conditional) { throw std::runtime_error(err_msg); }
#define VK_ASSERT(conditional, err_msg) if (conditional != VK_SUCCESS) { throw std::runtime_error(err_msg); }

// ----------------------------------------------------------------------------
//  Structs
// ----------------------------------------------------------------------------

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;

	bool isComplete()
	{
		return graphicsFamily.has_value();
	}
};


// ----------------------------------------------------------------------------
// Public Functions
// ----------------------------------------------------------------------------

QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device)
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
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			indices.graphicsFamily = i;
		}

		if (indices.isComplete())
			break;
	}

	return indices;
}

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
	const uint32_t wWidth = 1080;
	const uint32_t wHeight = 720;

	void initWindow()
	{
		glfwInit();                                      // Init GLFW lib

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);    // We are not using OpenGL
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);      // No window resize (for now)

		// Create window
		window = glfwCreateWindow(wWidth, wHeight, "Vulkan", nullptr, nullptr);
	}

	// ------------------------------------------------------------------------
	// Application API

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
		}
	}

	void cleanup()
	{
		// Destroy VK Debug Messenger
		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroyInstance(instance, nullptr);	// Destroy VK Instance

		glfwDestroyWindow(window);				// Destroy native window
		glfwTerminate();                        // De-init GLFW library
	}

	// ------------------------------------------------------------------------
	// Vulkan API

	VkInstance instance;								// Connection between application and the VK Library.
	VkDebugUtilsMessengerEXT debugMessenger;			// Debug Callback Messenger Handle
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;	// Physical Device (Graphics card) handle that we will be using

	// Validation layers are optional components that hook into Vulkan function calls to apply additional operations (Debugging).
	const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

	// Toggle validation layers.
#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		pickPhysicalDevice();
	}

	void pickPhysicalDevice()
	{
		// Get GPU count
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		ASSERT(deviceCount > 0, "Failed to find GPUs that support Vulkan!")

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

		// Physical device has to have atleast a graphics queue family to be supported
		QueueFamilyIndices indices = FindQueueFamilies(device);

		return indices.isComplete();
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
