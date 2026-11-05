import Foundation

/// Test-only builders for binary little-endian COLMAP `cameras.bin`, `images.bin` and
/// `points3D.bin` files. `count` overrides the leading `u64` record count (for corrupt input).

struct FixtureCamera {
    var id: Int32
    var model: Int32
    var width: UInt64
    var height: UInt64
    var params: [Double]
}

struct FixtureImage {
    var id: UInt32
    var rotation: SIMD4<Double>
    var translation: SIMD3<Double>
    var cameraID: UInt32
    var name: String
    var points2D: [(x: Double, y: Double, point3D: Int64)] = []
}

struct FixturePoint {
    var id: UInt64
    var position: SIMD3<Double>
    var color: SIMD3<UInt8>
    var error: Double
    var track: [(imageID: Int32, point2DIndex: Int32)] = []
}

func makeCamerasBinary(_ cameras: [FixtureCamera], count: UInt64? = nil) -> Data {
    var data = Data()
    colmapAppend(UInt64(count ?? UInt64(cameras.count)), to: &data)
    for camera in cameras {
        colmapAppend(camera.id, to: &data)
        colmapAppend(camera.model, to: &data)
        colmapAppend(camera.width, to: &data)
        colmapAppend(camera.height, to: &data)
        for value in camera.params {
            colmapAppend(value, to: &data)
        }
    }
    return data
}

func makeImagesBinary(_ images: [FixtureImage], count: UInt64? = nil) -> Data {
    var data = Data()
    colmapAppend(UInt64(count ?? UInt64(images.count)), to: &data)
    for image in images {
        colmapAppend(image.id, to: &data)
        for value in [image.rotation.x, image.rotation.y, image.rotation.z, image.rotation.w] {
            colmapAppend(value, to: &data)
        }
        for value in [image.translation.x, image.translation.y, image.translation.z] {
            colmapAppend(value, to: &data)
        }
        colmapAppend(image.cameraID, to: &data)
        data.append(contentsOf: Array(image.name.utf8))
        data.append(0)
        colmapAppend(UInt64(image.points2D.count), to: &data)
        for point in image.points2D {
            colmapAppend(point.x, to: &data)
            colmapAppend(point.y, to: &data)
            colmapAppend(point.point3D, to: &data)
        }
    }
    return data
}

func makePoints3DBinary(_ points: [FixturePoint], count: UInt64? = nil) -> Data {
    var data = Data()
    colmapAppend(UInt64(count ?? UInt64(points.count)), to: &data)
    for point in points {
        colmapAppend(point.id, to: &data)
        for value in [point.position.x, point.position.y, point.position.z] {
            colmapAppend(value, to: &data)
        }
        data.append(contentsOf: [point.color.x, point.color.y, point.color.z])
        colmapAppend(point.error, to: &data)
        colmapAppend(UInt64(point.track.count), to: &data)
        for entry in point.track {
            colmapAppend(entry.imageID, to: &data)
            colmapAppend(entry.point2DIndex, to: &data)
        }
    }
    return data
}

/// Appends `value` as little-endian bytes.
private func colmapAppend<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

private func colmapAppend(_ value: Double, to data: inout Data) {
    colmapAppend(value.bitPattern, to: &data)
}

private func colmapAppend(_ value: Int64, to data: inout Data) {
    colmapAppend(UInt64(bitPattern: value), to: &data)
}
