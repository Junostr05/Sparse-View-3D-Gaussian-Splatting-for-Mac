import Foundation
import SplatCore

/// Writes 3DGS PLY files, the inverse of `PLYReader`.
///
/// Output is `binary_little_endian 1.0` with float properties in the standard 3DGS order:
/// `x y z nx ny nz f_dc_0..2 f_rest_0..(3(K−1)−1) opacity scale_0..2 rot_0..3`.
/// `f_rest` is channel-major (`f_rest[c·(K−1) + (k−1)]`), matching `PLYReader`. Normals are 0.
public enum PLYWriter {
    public static func write(_ cloud: GaussianCloud) -> Data {
        let n = cloud.count
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        let restCount = 3 * (k - 1)

        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "element vertex \(n)\n"
        for name in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] {
            header += "property float \(name)\n"
        }
        for index in 0..<restCount {
            header += "property float f_rest_\(index)\n"
        }
        for name in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
            header += "property float \(name)\n"
        }
        header += "end_header\n"

        // 17 fixed floats per vertex (position, normal, DC, opacity, scale, rotation) plus f_rest.
        let floatsPerVertex = 17 + restCount
        var words: [UInt32] = []
        words.reserveCapacity(n * floatsPerVertex)
        for i in 0..<n {
            for c in 0..<3 {
                words.append(cloud.positions[3 * i + c].bitPattern.littleEndian)
            }
            for _ in 0..<3 {
                words.append(Float(0).bitPattern.littleEndian)
            }
            for c in 0..<3 {
                words.append(cloud.sh[(i * k) * 3 + c].bitPattern.littleEndian)
            }
            for c in 0..<3 {
                for coefficient in 1..<k {
                    words.append(cloud.sh[(i * k + coefficient) * 3 + c].bitPattern.littleEndian)
                }
            }
            words.append(cloud.opacityLogits[i].bitPattern.littleEndian)
            for c in 0..<3 {
                words.append(cloud.logScales[3 * i + c].bitPattern.littleEndian)
            }
            for q in 0..<4 {
                words.append(cloud.rotations[4 * i + q].bitPattern.littleEndian)
            }
        }

        var data = Data(header.utf8)
        data.append(words.withUnsafeBufferPointer { Data(buffer: $0) })
        return data
    }
}
