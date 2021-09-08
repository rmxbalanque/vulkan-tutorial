cmake_minimum_required(VERSION 3.13)
project(vulkan-tutorial)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        src/main.cpp
    )

add_executable(vulkan-tutorial ${SOURCE_FILES})

target_compile_definitions(vulkan-tutorial PRIVATE VK_PRJ_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Link with Vulkan.
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)

# Link with GLFW.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation." FORCE) # Turn off building GLFW documentation.
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests." FORCE)        # Turn off building GLFW tests.
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples." FORCE)  # Turn off building GLFW example code.
add_subdirectory(lib/glfw)
target_link_libraries(vulkan-tutorial glfw)

# Link with GLM.
add_subdirectory(lib/glm)
target_link_libraries(vulkan-tutorial glm)