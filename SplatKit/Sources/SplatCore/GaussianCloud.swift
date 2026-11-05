public enum GaussianCloudError: Error, Equatable {
    case invalidSHDegree(Int)
    case inconsistentLengths
}

/// Structure-of-arrays storage for a set of 3D Gaussians.
/// Layouts: positions 3n, logScales 3n, rotations 4n (w x y z), opacityLogits n,
/// sh n·K·3 where K = (shDegree + 1)².
public struct GaussianCloud: Sendable {
    public let shDegree: Int
    public let positions: [Float]
    public let logScales: [Float]
    public let rotations: [Float]
    public let opacityLogits: [Float]
    public let sh: [Float]

    public var count: Int { opacityLogits.count }

    public init(shDegree: Int,
                positions: [Float],
                logScales: [Float],
                rotations: [Float],
                opacityLogits: [Float],
                sh: [Float]) throws(GaussianCloudError) {
        guard (0...3).contains(shDegree) else {
            throw .invalidSHDegree(shDegree)
        }
        let n = opacityLogits.count
        let k = Self.coefficientCount(degree: shDegree)
        guard positions.count == 3 * n,
              logScales.count == 3 * n,
              rotations.count == 4 * n,
              sh.count == n * k * 3 else {
            throw .inconsistentLengths
        }
        self.shDegree = shDegree
        self.positions = positions
        self.logScales = logScales
        self.rotations = rotations
        self.opacityLogits = opacityLogits
        self.sh = sh
    }

    /// Unchecked initializer for data whose lengths are already known to be consistent.
    init(validated shDegree: Int,
                 positions: [Float],
                 logScales: [Float],
                 rotations: [Float],
                 opacityLogits: [Float],
                 sh: [Float]) {
        self.shDegree = shDegree
        self.positions = positions
        self.logScales = logScales
        self.rotations = rotations
        self.opacityLogits = opacityLogits
        self.sh = sh
    }

    /// Number of SH coefficients per color channel for `degree`: (degree + 1)².
    public static func coefficientCount(degree: Int) -> Int {
        (degree + 1) * (degree + 1)
    }

    /// Returns a copy without Gaussians that contain any NaN or infinite value,
    /// together with the number of dropped Gaussians.
    public func removingNonFinite() -> (cloud: GaussianCloud, removed: Int) {
        let k = Self.coefficientCount(degree: shDegree)
        var keptPositions: [Float] = []
        var keptLogScales: [Float] = []
        var keptRotations: [Float] = []
        var keptOpacity: [Float] = []
        var keptSH: [Float] = []
        var removed = 0

        for i in 0..<count {
            let pos = positions[3 * i..<3 * i + 3]
            let scl = logScales[3 * i..<3 * i + 3]
            let rot = rotations[4 * i..<4 * i + 4]
            let opacity = opacityLogits[i]
            let shSlice = sh[i * k * 3..<(i + 1) * k * 3]

            let allFinite = pos.allSatisfy(\.isFinite)
                && scl.allSatisfy(\.isFinite)
                && rot.allSatisfy(\.isFinite)
                && opacity.isFinite
                && shSlice.allSatisfy(\.isFinite)

            guard allFinite else {
                removed += 1
                continue
            }
            keptPositions.append(contentsOf: pos)
            keptLogScales.append(contentsOf: scl)
            keptRotations.append(contentsOf: rot)
            keptOpacity.append(opacity)
            keptSH.append(contentsOf: shSlice)
        }

        let cloud = GaussianCloud(
            validated: shDegree,
            positions: keptPositions,
            logScales: keptLogScales,
            rotations: keptRotations,
            opacityLogits: keptOpacity,
            sh: keptSH
        )
        return (cloud, removed)
    }
}
