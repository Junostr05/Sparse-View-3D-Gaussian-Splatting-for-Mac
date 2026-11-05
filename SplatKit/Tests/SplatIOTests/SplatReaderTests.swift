import Foundation
import Testing
import SplatCore
import SplatIO

/// Test-only builder for one 32-byte `.splat` record (little-endian).
private func splatRecord(position: SIMD3<Float>,
                         scale: SIMD3<Float>,
                         rgba: [UInt8],
                         rotation: [UInt8]) -> Data {
    var data = Data()
    for value in [position.x, position.y, position.z, scale.x, scale.y, scale.z] {
        withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
    }
    data.append(contentsOf: rgba)
    data.append(contentsOf: rotation)
    return data
}

@Test func decodesRecord() throws {
    let data = splatRecord(position: SIMD3(1, 2, 3),
                           scale: SIMD3(0.5, 0.5, 0.5),
                           rgba: [255, 128, 0, 255],
                           rotation: [255, 128, 128, 128])

    let cloud = try AntimatterSplatReader.read(data)

    #expect(cloud.shDegree == 0)
    #expect(cloud.count == 1)
    #expect(approxEqual(cloud.positions, [1, 2, 3], tol: 1e-4))
    #expect(approxEqual(cloud.logScales, [-0.693147, -0.693147, -0.693147], tol: 1e-4))
    #expect(approxEqual(cloud.sh, [1.772454, 0.006958, -1.772454], tol: 1e-4))
    #expect(approxEqual(cloud.opacityLogits, [9.21024], tol: 1e-4))
    // Decoded quaternion is (127/128, 0, 0, 0); the reader normalizes it to (1, 0, 0, 0).
    #expect(approxEqual(cloud.rotations, [1, 0, 0, 0], tol: 1e-4))
}

@Test func rejectsSizeNotMultipleOf32() {
    #expect(throws: SplatIOError.invalidSize(33)) {
        try AntimatterSplatReader.read(Data(count: 33))
    }
}

@Test func rejectsEmpty() {
    #expect(throws: SplatIOError.empty) {
        try AntimatterSplatReader.read(Data())
    }
}

@Test func sceneLoaderDispatchesByExtension() throws {
    let props = standardProperties(restCount: 0)
    let ply = makePLY(properties: props, rows: [
        makeRow(props, ["x": 1, "y": 2, "z": 3, "scale_0": 0, "scale_1": 0, "scale_2": 0,
                        "rot_0": 1, "opacity": 0]),
    ])
    let plyScene = try SceneLoader.load(ply, fileExtension: "PLY")
    #expect(plyScene.cloud.count == 1)
    #expect(plyScene.cloud.positions == [1, 2, 3])

    let splat = splatRecord(position: SIMD3(4, 5, 6),
                            scale: SIMD3(1, 1, 1),
                            rgba: [0, 0, 0, 255],
                            rotation: [255, 128, 128, 128])
    let splatScene = try SceneLoader.load(splat, fileExtension: "splat")
    #expect(splatScene.cloud.count == 1)
    #expect(splatScene.cloud.positions == [4, 5, 6])

    #expect(throws: SplatIOError.unsupportedFormat("obj")) {
        try SceneLoader.load(Data(), fileExtension: "obj")
    }
}

@Test func sceneLoaderReportsRemovedCount() throws {
    let bad = splatRecord(position: SIMD3(Float.nan, 0, 0),
                          scale: SIMD3(1, 1, 1),
                          rgba: [0, 0, 0, 255],
                          rotation: [255, 128, 128, 128])
    let good = splatRecord(position: SIMD3(7, 8, 9),
                           scale: SIMD3(1, 1, 1),
                           rgba: [0, 0, 0, 255],
                           rotation: [255, 128, 128, 128])

    let scene = try SceneLoader.load(bad + good, fileExtension: "splat")

    #expect(scene.removedNonFinite == 1)
    #expect(scene.cloud.count == 1)
    #expect(scene.cloud.positions == [7, 8, 9])
}

@Test func sceneLoaderThrowsEmptyWhenAllNonFinite() {
    let bad = splatRecord(position: SIMD3(0, Float.infinity, 0),
                          scale: SIMD3(1, 1, 1),
                          rgba: [0, 0, 0, 255],
                          rotation: [255, 128, 128, 128])

    #expect(throws: SplatIOError.empty) {
        try SceneLoader.load(bad, fileExtension: "splat")
    }
}
