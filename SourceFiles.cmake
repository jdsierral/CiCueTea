set(IncludeFiles
)

set(SourceFiles
    Source/CQT.cpp
    Source/CQT.hpp
    Source/FFT.cpp
    Source/FFT.hpp
    Source/Slicer.cpp
    Source/Slicer.hpp
    Source/Splicer.cpp
    Source/Splicer.hpp
    Source/RingBuffer.cpp
    Source/RingBuffer.hpp
    Source/OverlapAddProcessor.cpp
    Source/OverlapAddProcessor.hpp
    Source/Windows.h
    Source/VectorOps.h
    Source/Benchtools.h
)

set(SourceFiles
    ${IncludeFiles}
    ${SourceFiles}
)

source_group("Source Files\\Source" FILES ${SourceFiles})