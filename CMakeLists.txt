cmake_minimum_required(VERSION 3.24)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NUMERICAL_TOOLBOX_STANDALONE On)
endif()

option(CMAKE_COMPILE_WARNING_AS_ERROR "Enable warnings-as-error" On)
option(NUMERICAL_TOOLBOX_INCLUDE_DEFAULT_INIT "Include default initialization code; turn off when providing custom initialization" On)
option(NUMERICAL_TOOLBOX_BUILD_EXAMPLES "Enable build of the examples" Off)
option(NUMERICAL_TOOLBOX_BUILD_TESTS "Enable build of the tests" Off)

if (NUMERICAL_TOOLBOX_BUILD_TESTS)
    # CTest cannot be included before the first project() statement, but Embedded Infastructure
    # Library needs to see that test utilities need to be built.
    set(BUILD_TESTING On)
endif()

add_definitions(-DEMIL_ENABLE_TRACING=1)

if (NUMERICAL_TOOLBOX_STANDALONE)
    include(FetchContent)

    FetchContent_Declare(
        emil
        GIT_REPOSITORY https://github.com/embedded-pro/embedded-infra-lib.git
        GIT_TAG        3a172983dcfe206c6d4c7109175022597966dbdd # Unreleased
    )

    FetchContent_MakeAvailable(emil)
endif()

project(numerical LANGUAGES C CXX ASM VERSION 0.0.1) # x-release-please-version

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)

if (NUMERICAL_TOOLBOX_STANDALONE)
    FetchContent_Declare(
        sciplot
        GIT_REPOSITORY          https://github.com/sciplot/sciplot.git
        GIT_TAG                 v0.2.2
        GIT_SHALLOW             TRUE
        GIT_SUBMODULES_RECURSE  TRUE
    )
    
    FetchContent_GetProperties(sciplot)
    if(NOT sciplot_POPULATED)
        FetchContent_Populate(sciplot)
    endif()

    add_library(sciplot INTERFACE)
    target_include_directories(sciplot INTERFACE 
        ${sciplot_SOURCE_DIR}
        OPTIONS 
        -Wno-error           # Disable -Werror for sciplot headers
        -Wno-unused-variable # Disable specific warning
        -Wno-sign-compare    # Disable specific warning
    )
endif()

set_directory_properties(PROPERTY USE_FOLDERS ON)

# When building usb-host by itself do not exclude any targets from all
# Libraries will only be excluded from all when usb-host is a consumed as a dependency.
if (NUMERICAL_TOOLBOX_STANDALONE)
    set(NUMERICAL_TOOLBOX_EXCLUDE_FROM_ALL "")
else()
    set(NUMERICAL_TOOLBOX_EXCLUDE_FROM_ALL "EXCLUDE_FROM_ALL")
endif()

if (NUMERICAL_TOOLBOX_BUILD_TESTS)
    include(CTest)
    emil_enable_testing()
endif()

add_subdirectory(examples)
add_subdirectory(numerical)

emil_clangformat_directories(examples DIRECTORIES .)
emil_clangformat_directories(numerical DIRECTORIES .)

if (NUMERICAL_TOOLBOX_STANDALONE)
    emil_folderize_all_targets()
endif()
