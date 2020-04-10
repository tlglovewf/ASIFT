cmake_minimum_required(VERSION 2.8.4)
project(Erie)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set( Boost_USE_STATIC_LIBS OFF )
set( Boost_USE_STATIC_RUNTIME OFF )
find_package( Boost REQUIRED COMPONENTS system filesystem )
set(OpenCV_DIR /usr/local/share/OpenCV)
#ex find opencv 
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()


#set(OpenCV_LIBS /usr/local/lib)

include_directories(src)



set(TCLAP_HEADERS
    src/tclap/Arg.h
    src/tclap/ArgException.h
    src/tclap/ArgTraits.h
    src/tclap/CmdLine.h
    src/tclap/CmdLineInterface.h
    src/tclap/CmdLineOutput.h
    src/tclap/Constraint.h
    src/tclap/DocBookOutput.h
    src/tclap/HelpVisitor.h
    src/tclap/IgnoreRestVisitor.h
    src/tclap/MultiArg.h
    src/tclap/MultiSwitchArg.h
    src/tclap/OptionalUnlabeledTracker.h
    src/tclap/StandardTraits.h
    src/tclap/StdOutput.h
    src/tclap/SwitchArg.h
    src/tclap/UnlabeledMultiArg.h
    src/tclap/UnlabeledValueArg.h
    src/tclap/ValueArg.h
    src/tclap/ValuesConstraint.h
    src/tclap/VersionVisitor.h
    src/tclap/Visitor.h
    src/tclap/XorHandler.h
    src/tclap/ZshCompletionOutput.h)

set(UTILITIES_HEADERS
    src/apps/featuresIO.cpp
    src/apps/featuresIO.h
    src/apps/mediaIO.cpp
    src/apps/mediaIO.h)

set(ERIE_SRC_FILES
    src/aff_angles.cpp
    src/aff_features2d.cpp
    src/aff_features2d.hpp
    src/aff_helper.cpp
    src/aff_matchers.cpp)

add_library( erie ${ERIE_SRC_FILES} )

message(STATUS ${OpenCV_LIBS})

add_executable(aff_demo         src/apps/aff_demo.cpp         src/aff_features2d.hpp ${UTILITIES_HEADERS})
add_executable(aff_match_images src/apps/aff_match_images.cpp src/aff_features2d.hpp ${UTILITIES_HEADERS} ${TCLAP_HEADERS})
add_executable(aff_match_video  src/apps/aff_match_video.cpp  src/aff_features2d.hpp ${UTILITIES_HEADERS} ${TCLAP_HEADERS})

target_link_libraries( aff_demo         erie ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )
target_link_libraries( aff_match_images erie ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )
target_link_libraries( aff_match_video  erie ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )
