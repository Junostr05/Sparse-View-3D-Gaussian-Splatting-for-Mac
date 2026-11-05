import Foundation
import Metal
import SplatCore
@testable import SplatMetal

/// Deterministic SplitMix64 generator.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum TestEnvironment {
    /// Prints the Metal device name once so CI logs show whether Metal tests really ran.
    static let deviceName: String = {
        let name = MTLCreateSystemDefaultDevice()?.name ?? "none"
        print("SplatMetalTests: Metal device = \(name)")
        return name
    }()

    static var hasMetal: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
}

func makeTestContext() throws -> MetalContext {
    _ = TestEnvironment.deviceName
    return try MetalContext.makeDefault()
}

func approxEqual(_ a: Float, _ b: Float, tol: Float) -> Bool {
    abs(a - b) <= tol
}

/// Random scene in front of a camera with R = I, t = 0 (camera space z in `depth`).
func makeRandomCloud(
    count: Int,
    seed: UInt64 = 42,
    shDegree: Int = 3,
    depth: ClosedRange<Float> = 2...8,
    scale: ClosedRange<Float> = 0.02...0.3
) throws -> GaussianCloud {
    var rng = SplitMix64(seed: seed)
    let k = GaussianCloud.coefficientCount(degree: shDegree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    positions.reserveCapacity(3 * count)
    logScales.reserveCapacity(3 * count)
    rotations.reserveCapacity(4 * count)
    opacity.reserveCapacity(count)
    sh.reserveCapacity(count * k * 3)

    let logLo = log(scale.lowerBound)
    let logHi = log(scale.upperBound)
    for _ in 0..<count {
        let z = Float.random(in: depth, using: &rng)
        let x = Float.random(in: -0.45...0.45, using: &rng) * abs(z)
        let y = Float.random(in: -0.45...0.45, using: &rng) * abs(z)
        positions += [x, y, z]
        for _ in 0..<3 {
            logScales.append(Float.random(in: logLo...logHi, using: &rng))
        }
        for _ in 0..<4 {
            rotations.append(Float.random(in: -1...1, using: &rng))
        }
        opacity.append(Float.random(in: -2...4, using: &rng))
        for coefficient in 0..<k {
            for _ in 0..<3 {
                if coefficient == 0 {
                    sh.append(Float.random(in: 0...1.5, using: &rng))
                } else {
                    sh.append(Float.random(in: -0.5...0.5, using: &rng))
                }
            }
        }
    }
    return try GaussianCloud(shDegree: shDegree, positions: positions, logScales: logScales,
                             rotations: rotations, opacityLogits: opacity, sh: sh)
}

/// Pinhole camera with R = I, t = 0, fx = fy = height, principal point at the image center.
func makeTestCamera(width: Int, height: Int) -> Camera {
    Camera(width: width, height: height,
           fx: Float(height), fy: Float(height),
           cx: Float(width) / 2, cy: Float(height) / 2,
           rotation: .identity, translation: .zero)
}
