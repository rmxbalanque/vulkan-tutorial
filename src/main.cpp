
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <vector>

#define VK_ASSERT(conditional, err_msg) if (conditional != VK_SUCCESS) { throw std::runtime_error(err_msg); }

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
		glfwInit();                                        // Init GLFW lib

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);    // We are not using OpenGL
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);        // No window resize (for now)

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
		vkDestroyInstance(instance, nullptr);    // Destroy VK Instance

		glfwDestroyWindow(window);                        // Destroy native window
		glfwTerminate();                                // Deinit GLFW library
	}

	// ------------------------------------------------------------------------
	// Vulkan API

	// Connection between application and the VK Library.
	VkInstance instance;

	void initVulkan()
	{
		createInstance();
	}

	void createInstance()
	{
		// Setup application info
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// Get GLFW required extensions from Vulkan to handle the native window
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		// Setup creation info
		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;
		createInfo.enabledLayerCount = 0;

		// Create VK Instance
		VK_ASSERT(vkCreateInstance(&createInfo, nullptr, &instance), "Failed to create instance");

		// Show available and supported extensions
		displayExtensionsInfo();
	}

	void displayExtensionsInfo()
	{
		// Get GLFW required extensions from Vulkan to handle the native window
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		// Check for supported VK Extensions
		uint32_t extensionCount = 0;
		VK_ASSERT(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "Failed to get supported extensions count");

		std::vector<VkExtensionProperties> extensions(extensionCount);
		VK_ASSERT(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data()), "Failed to get supported extensions properties");

		// Log required GLFW Extensions
		std::cout << "GLFW Vulkan extensions:\n";
		for (uint32_t i = 0; i < glfwExtensionCount; ++i)
		{
			uint8_t isSupported = 0;

			// Check GLFW Extension is supported
			for (const auto & extension : extensions)
			{
				if (std::strcmp(extension.extensionName, glfwExtensions[i]) == 0)
				{
					isSupported = 1;
					break;
				}
			}

			std::cout << '\t'  << "- " << glfwExtensions[i] << " | " << (isSupported ? "SUPPORTED" : "NOT SUPPORTED") << '\n';
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
