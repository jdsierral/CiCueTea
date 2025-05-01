set(IncludeFiles
    Include/RingBuffer.hpp
    Include/Slicer.hpp
    Include/Splicer.hpp
    Include/FFT.hpp
    Include/CQT.hpp
    Include/CQTProcessor.hpp
    Include/MathUtils.h
    Include/SignalUtils.h
)

set(SourceFiles
    Source/FFT_FFTW.h
    Source/FFT_vDSP.h
    Source/FFT_PFFFT.h
    Source/VectorOps.h
    Source/Benchtools.h
    Source/RingBuffer.cpp
    Source/Splicer.cpp
    Source/Slicer.cpp
    Source/FFT.cpp
    Source/CQT.cpp
    Source/CQTProcessor.cpp
)

set(SourceFiles
    ${IncludeFiles}
    ${SourceFiles}
)

source_group("Source Files\\Source" FILES ${SourceFiles})