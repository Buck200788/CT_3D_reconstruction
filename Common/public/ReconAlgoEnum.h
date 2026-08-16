#pragma once

enum class ReconAlgorithm
{
    FDK,
    Katsevich
};

inline std::string ReconAlgorithmToString(ReconAlgorithm alg)
{
    switch (alg)
    {
    case ReconAlgorithm::FDK:
        return "FDK";
    case ReconAlgorithm::Katsevich:
        return "Katsevich";
    default:
        return "UnknownReconAlgorithm";
    }
}

// string -> enum
inline ReconAlgorithm StringToReconAlgorithm(const std::string& str)
{
    if (str == "FDK")      return ReconAlgorithm::FDK;
    if (str == "Katsevich") return ReconAlgorithm::Katsevich;
    return static_cast<ReconAlgorithm>(-1);
}