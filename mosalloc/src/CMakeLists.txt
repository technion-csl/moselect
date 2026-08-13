file(GLOB SRCS "*.cc")
file(GLOB HDRS "${CMAKE_SOURCE_DIR}/include/*.h")

add_library(${PROJECT_NAME} SHARED ${SRCS} ${HDRS})
target_link_libraries(${PROJECT_NAME} dl)
target_include_directories(${PROJECT_NAME} PRIVATE ../include)

# The following workaround is to prevent building the test apps with hooks.cc.
# If the test apps are linked with hooks.cc, then the constructor is called at runtime,
# which overrides the memory functions, reads the environment variables, etc.
# The test apps don't need the constructor, they only need the API of the mosalloc classes.
list(FILTER SRCS EXCLUDE REGEX ".*/hooks.cc")
# drop out hooks.h and malloc.cc (dlmalloc source) as they are not needed for tests
list(FILTER SRCS EXCLUDE REGEX ".*/malloc.cc")
list(FILTER HDRS EXCLUDE REGEX ".*/hooks.h")

add_library(${API_LIBRARY} SHARED ${SRCS} ${HDRS})
target_link_libraries(${API_LIBRARY} dl)
message(STATUS "api-library: ${API_LIBRARY}")
target_include_directories(${API_LIBRARY} PUBLIC ../include)
