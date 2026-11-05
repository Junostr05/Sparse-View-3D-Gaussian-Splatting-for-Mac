/// Errors thrown by the SplatIO readers.
public enum SplatIOError: Error, Equatable {
    /// The data is not a PLY file, or its header is malformed or incomplete.
    case notPLY
    /// The file uses a feature this reader does not support. The payload names it.
    case unsupportedFormat(String)
    /// A property required for a Gaussian is absent from the header.
    case missingProperty(String)
    /// The number of `f_rest_*` properties does not match any supported SH degree.
    case unsupportedSHCount(Int)
    /// The vertex body is shorter than the header declares. Both values are byte counts.
    case truncated(expected: Int, actual: Int)
    /// A count in the header is negative or the body size overflows.
    case invalidSize(Int)
    /// The file declares zero vertices.
    case empty
}
