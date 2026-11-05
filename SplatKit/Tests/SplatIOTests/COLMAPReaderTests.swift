import Foundation
import Testing
import SplatCore
@testable import SplatIO

private let allModelIDs: [Int32] = Array(0...11)

@Test func readsCamerasOfEverySupportedModel() throws {
    let cameras = allModelIDs.map { model in
        FixtureCamera(id: model + 1,
                      model: model,
                      width: 640 + UInt64(model),
                      height: 480,
                      params: (0..<COLMAPReader.parameterCount(modelID: model)!).map { Double($0) + 0.25 })
    }
    let parsed = try COLMAPReader.readCameras(makeCamerasBinary(cameras))
    #expect(parsed.count == cameras.count)
    for (camera, expected) in zip(parsed, cameras) {
        #expect(camera.id == UInt32(expected.id))
        #expect(camera.modelID == expected.model)
        #expect(camera.width == Int(expected.width))
        #expect(camera.height == Int(expected.height))
        #expect(camera.params == expected.params)
    }
}

@Test func parameterCountsMatchCOLMAPModels() {
    let counts: [Int32: Int] = [0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16]
    for (model, count) in counts {
        #expect(COLMAPReader.parameterCount(modelID: model) == count)
    }
    #expect(COLMAPReader.parameterCount(modelID: 12) == nil)
    #expect(COLMAPReader.parameterCount(modelID: -1) == nil)
}

@Test func readsImagesWithPoints2DAndUTF8Names() throws {
    let images = [
        FixtureImage(id: 7,
                     rotation: SIMD4(1, 0, 0, 0),
                     translation: SIMD3(0.5, -1, 2),
                     cameraID: 1,
                     name: "cam0/ä.jpg",
                     points2D: [(x: 1, y: 2, point3D: 10),
                                (x: 3, y: 4, point3D: -1),
                                (x: 5, y: 6, point3D: 11)]),
        FixtureImage(id: 3,
                     rotation: SIMD4(0.5, 0.5, -0.5, 0.5),
                     translation: SIMD3(-2, 0, 1),
                     cameraID: 2,
                     name: "a.png"),
    ]
    let parsed = try COLMAPReader.readImages(makeImagesBinary(images))
    // Sorted by name: "a.png" sorts before "cam0/ä.jpg".
    #expect(parsed.map(\.name) == ["a.png", "cam0/ä.jpg"])
    let second = parsed[1]
    #expect(second.id == 7)
    #expect(second.rotation == SIMD4(1, 0, 0, 0))
    #expect(second.translation == SIMD3(0.5, -1, 2))
    #expect(second.cameraID == 1)
    let first = parsed[0]
    #expect(first.id == 3)
    #expect(first.rotation == SIMD4(0.5, 0.5, -0.5, 0.5))
    #expect(first.translation == SIMD3(-2, 0, 1))
    #expect(first.cameraID == 2)
}

@Test func readsPoints3DWithTracks() throws {
    let points = [
        FixturePoint(id: 10,
                     position: SIMD3(1, 2, 3),
                     color: SIMD3(255, 128, 0),
                     error: 0.5,
                     track: [(imageID: 1, point2DIndex: 0), (imageID: 2, point2DIndex: 4)]),
        FixturePoint(id: 11,
                     position: SIMD3(-1, 0.25, 9),
                     color: SIMD3(1, 2, 3),
                     error: 0,
                     track: []),
    ]
    let parsed = try COLMAPReader.readPoints3D(makePoints3DBinary(points))
    #expect(parsed == [
        COLMAPPoint(position: SIMD3(1, 2, 3), color: SIMD3(255, 128, 0)),
        COLMAPPoint(position: SIMD3(-1, 0.25, 9), color: SIMD3(1, 2, 3)),
    ])
}

@Test func truncatedFilesThrow() {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 4, height: 4, params: [1, 2, 3, 4]),
        FixtureCamera(id: 2, model: 0, width: 4, height: 4, params: [1, 2, 3]),
    ])
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 0),
                     cameraID: 1, name: "x.jpg",
                     points2D: [(x: 1, y: 2, point3D: 3)]),
        FixtureImage(id: 2, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 0),
                     cameraID: 1, name: "yy.jpg"),
    ])
    let points = makePoints3DBinary([
        FixturePoint(id: 1, position: SIMD3(0, 0, 0), color: SIMD3(0, 0, 0), error: 0,
                     track: [(imageID: 1, point2DIndex: 2)]),
        FixturePoint(id: 2, position: SIMD3(1, 1, 1), color: SIMD3(9, 9, 9), error: 1),
    ])

    for cut in 0..<cameras.count {
        #expect(throws: DatasetError.truncated(file: "cameras.bin")) {
            try COLMAPReader.readCameras(Data(cameras.prefix(cut)))
        }
    }
    for cut in 0..<images.count {
        #expect(throws: DatasetError.truncated(file: "images.bin")) {
            try COLMAPReader.readImages(Data(images.prefix(cut)))
        }
    }
    for cut in 0..<points.count {
        #expect(throws: DatasetError.truncated(file: "points3D.bin")) {
            try COLMAPReader.readPoints3D(Data(points.prefix(cut)))
        }
    }
}

@Test func hugeCountIsRejectedWithoutAllocation() {
    let huge = UInt64.max
    #expect(throws: DatasetError.truncated(file: "cameras.bin")) {
        try COLMAPReader.readCameras(makeCamerasBinary([], count: huge))
    }
    #expect(throws: DatasetError.truncated(file: "images.bin")) {
        try COLMAPReader.readImages(makeImagesBinary([], count: huge))
    }
    #expect(throws: DatasetError.truncated(file: "points3D.bin")) {
        try COLMAPReader.readPoints3D(makePoints3DBinary([], count: huge))
    }
}

@Test func hugeNestedCountsAreRejected() {
    // A 2D point count and a track length that overflow the remaining bytes.
    var images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 0),
                     cameraID: 1, name: "a.jpg"),
    ])
    // n2D sits 8 bytes before the end of this record (no 2D points follow).
    images.replaceSubrange((images.count - 8)..<images.count,
                           with: withUnsafeBytes(of: UInt64.max.littleEndian) { Array($0) })
    #expect(throws: DatasetError.truncated(file: "images.bin")) {
        try COLMAPReader.readImages(images)
    }

    var points = makePoints3DBinary([
        FixturePoint(id: 1, position: SIMD3(0, 0, 0), color: SIMD3(0, 0, 0), error: 0),
    ])
    points.replaceSubrange((points.count - 8)..<points.count,
                           with: withUnsafeBytes(of: UInt64.max.littleEndian) { Array($0) })
    #expect(throws: DatasetError.truncated(file: "points3D.bin")) {
        try COLMAPReader.readPoints3D(points)
    }
}

@Test func unterminatedImageNameIsTruncated() {
    var data = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 0),
                     cameraID: 1, name: "abc"),
    ])
    // Drop the terminator and the 2D point count so the name runs to the end of the file.
    data.removeLast(9)
    #expect(throws: DatasetError.truncated(file: "images.bin")) {
        try COLMAPReader.readImages(data)
    }
}

@Test func unknownModelThrows() throws {
    let data = makeCamerasBinary([FixtureCamera(id: 1, model: 99, width: 4, height: 4, params: [])])
    #expect(throws: DatasetError.unknownCameraModel(99)) {
        try COLMAPReader.readCameras(data)
    }
}

@Test func fisheyeModelIsUnsupported() throws {
    let camera = COLMAPCamera(id: 1, modelID: 5, width: 4, height: 4, params: Array(repeating: 1, count: 8))
    #expect(throws: DatasetError.unsupportedCameraModel("OPENCV_FISHEYE")) {
        try CameraModel(colmap: camera)
    }
}

@Test func supportedModelsConvertToCameraModel() throws {
    let simple = try CameraModel(colmap: COLMAPCamera(id: 1, modelID: 0, width: 4, height: 4,
                                                      params: [10, 2, 3]))
    #expect(simple == .simplePinhole(f: 10, cx: 2, cy: 3))

    let pinhole = try CameraModel(colmap: COLMAPCamera(id: 1, modelID: 1, width: 4, height: 4,
                                                       params: [10, 11, 2, 3]))
    #expect(pinhole == .pinhole(fx: 10, fy: 11, cx: 2, cy: 3))

    let simpleRadial = try CameraModel(colmap: COLMAPCamera(id: 1, modelID: 2, width: 4, height: 4,
                                                            params: [10, 2, 3, 0.1]))
    #expect(simpleRadial == .simpleRadial(f: 10, cx: 2, cy: 3, k: 0.1))

    let radial = try CameraModel(colmap: COLMAPCamera(id: 1, modelID: 3, width: 4, height: 4,
                                                      params: [10, 2, 3, 0.1, 0.2]))
    #expect(radial == .radial(f: 10, cx: 2, cy: 3, k1: 0.1, k2: 0.2))

    let opencv = try CameraModel(colmap: COLMAPCamera(id: 1, modelID: 4, width: 4, height: 4,
                                                      params: [10, 11, 2, 3, 0.1, 0.2, 0.3, 0.4]))
    #expect(opencv == .opencv(fx: 10, fy: 11, cx: 2, cy: 3, k1: 0.1, k2: 0.2, p1: 0.3, p2: 0.4))
}

@Test func unknownModelIDCannotConvertToCameraModel() {
    let camera = COLMAPCamera(id: 1, modelID: 42, width: 4, height: 4, params: [])
    #expect(throws: DatasetError.unknownCameraModel(42)) {
        try CameraModel(colmap: camera)
    }
}
