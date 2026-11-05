import Foundation

/// Sequential deterministic generator (SplitMix64). Library code never uses
/// `SystemRandomNumberGenerator`; all sequential randomness goes through this type.
public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Index-addressable hash-based randomness. The same integer arithmetic is mirrored in MSL
/// so CPU and GPU produce identical streams.
public enum SplatRandom {
    /// PCG-RXS-M-XS style 32-bit hash.
    public static func hash(_ x: UInt32) -> UInt32 {
        let s = x &* 747_796_405 &+ 2_891_336_453
        let w = ((s >> ((s >> 28) &+ 4)) ^ s) &* 277_803_737
        return (w >> 22) ^ w
    }

    /// Uniform sample strictly inside (0, 1).
    public static func uniform(seed: UInt32, index: UInt32, stream: UInt32) -> Float {
        let key = hash(seed ^ hash(index ^ hash(stream)))
        let value = (Float(key >> 8) + 0.5) / 16_777_216
        // Float(2^24 - 1) + 0.5 is not representable and ties to 2^24, which would give exactly 1.
        return min(value, 1 - Float.ulpOfOne / 2)
    }

    /// Standard normal sample (Box-Muller) using streams `2·stream` and `2·stream + 1`.
    public static func normal(seed: UInt32, index: UInt32, stream: UInt32) -> Float {
        let u1 = uniform(seed: seed, index: index, stream: stream &* 2)
        let u2 = uniform(seed: seed, index: index, stream: stream &* 2 &+ 1)
        return (-2 * log(u1)).squareRoot() * cos(2 * Float.pi * u2)
    }
}
