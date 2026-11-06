import Foundation

/// Writes `PoseEstimationResult` data as COLMAP binary files, the layout `COLMAPReader` reads.
///
/// All values are little-endian. Camera, image and point ids are 1-based indices into the arrays
/// passed in. Image 2D observations and point tracks are always empty.
public enum COLMAPWriter {
    /// `cameras.bin`. Camera `i` gets id `i + 1`.
    public static func cameras(_ cameras: [EstimatedCamera]) -> Data {
        var data = Data()
        appendLittleEndian(UInt64(cameras.count), to: &data)
        for (index, camera) in cameras.enumerated() {
            appendLittleEndian(UInt32(index + 1), to: &data)
            appendLittleEndian(camera.modelID, to: &data)
            appendLittleEndian(UInt64(camera.width), to: &data)
            appendLittleEndian(UInt64(camera.height), to: &data)
            for value in camera.params {
                appendLittleEndian(value.bitPattern, to: &data)
            }
        }
        return data
    }

    /// `images.bin`. Image `i` gets id `i + 1`. Its camera id is `cameraIndex + 1`.
    public static func images(_ images: [EstimatedImagePose]) -> Data {
        var data = Data()
        appendLittleEndian(UInt64(images.count), to: &data)
        for (index, image) in images.enumerated() {
            appendLittleEndian(UInt32(index + 1), to: &data)
            // Quaternion in (w, x, y, z) order, matching the reader.
            let rotation: [Double] = [image.rotation.x, image.rotation.y, image.rotation.z, image.rotation.w]
            for value in rotation {
                appendLittleEndian(value.bitPattern, to: &data)
            }
            let translation: [Double] = [image.translation.x, image.translation.y, image.translation.z]
            for value in translation {
                appendLittleEndian(value.bitPattern, to: &data)
            }
            appendLittleEndian(UInt32(image.cameraIndex + 1), to: &data)
            data.append(contentsOf: Array(image.name.utf8))
            data.append(0)
            appendLittleEndian(UInt64(0), to: &data)
        }
        return data
    }

    /// `points3D.bin`. Point `i` gets id `i + 1`.
    public static func points3D(_ points: [EstimatedPoint]) -> Data {
        var data = Data()
        appendLittleEndian(UInt64(points.count), to: &data)
        for (index, point) in points.enumerated() {
            appendLittleEndian(UInt64(index + 1), to: &data)
            let position: [Double] = [point.position.x, point.position.y, point.position.z]
            for value in position {
                appendLittleEndian(value.bitPattern, to: &data)
            }
            data.append(contentsOf: [point.color.x, point.color.y, point.color.z])
            appendLittleEndian(point.error.bitPattern, to: &data)
            appendLittleEndian(UInt64(0), to: &data)
        }
        return data
    }

    /// Appends the little-endian bytes of a fixed-width integer.
    private static func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
}
