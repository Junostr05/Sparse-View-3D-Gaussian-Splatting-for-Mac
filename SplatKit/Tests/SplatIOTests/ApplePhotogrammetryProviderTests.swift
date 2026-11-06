#if canImport(RealityKit) && canImport(ImageIO) && canImport(CoreGraphics)
import CoreGraphics
import Foundation
import ImageIO
import SplatCore
import Testing
@testable import SplatIO

// MARK: - Fixtures

/// A real PNG of the given pixel size, so ImageIO can read its dimensions.
private func pngData(width: Int, height: Int) -> Data {
    let space = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: nil,
                            width: width,
                            height: height,
                            bitsPerComponent: 8,
                            bytesPerRow: 0,
                            space: space,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let image = context.makeImage()!
    let output = NSMutableData()
    let destination = CGImageDestinationCreateWithData(output, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
    return output as Data
}

private func photo(_ name: String,
                   width: Int = 90,
                   height: Int = 120,
                   focal35mm: Double? = nil) -> CapturedFrame {
    CapturedFrame(name: name,
                  data: pngData(width: width, height: height),
                  sourceName: name,
                  timestamp: nil,
                  focalLength35mm: focal35mm)
}

private typealias ProviderPose = ApplePhotogrammetryPoseProvider.ProviderPose

private func pose(_ name: String,
                  centre: SIMD3<Double> = SIMD3<Double>(0, 0, 0),
                  intrinsics: SIMD4<Double>? = nil) -> ProviderPose {
    (name: name, centre: centre, quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), intrinsics: intrinsics)
}

/// Apple intrinsics (fx, fy, cx, cy) that are valid for a 90 x 120 frame.
private let validIntrinsics = SIMD4<Double>(1194.76, 1194.76, 45.2, 59.9)

// MARK: - buildResult

@Test func buildResultSharesOneCameraForEqualSizeAndFocal() throws {
    let frames = [photo("a.png", focal35mm: 26), photo("b.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("b.png"), pose("a.png")],
        points: [],
        fallbackFocalLength35mm: nil)

    #expect(result.cameras.count == 1)
    #expect(result.cameras[0].modelID == 0)
    #expect(result.images.map(\.name) == ["a.png", "b.png"])
    #expect(result.images.map(\.cameraIndex) == [0, 0])
}

@Test func buildResultUsesTwoCamerasForDifferentSizes() throws {
    let frames = [photo("a.png", width: 90, height: 120, focal35mm: 26),
                  photo("b.png", width: 120, height: 90, focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png"), pose("b.png")],
        points: [],
        fallbackFocalLength35mm: nil)

    #expect(result.cameras.count == 2)
    #expect(result.cameras.map(\.width) == [90, 120])
    #expect(result.cameras.map(\.height) == [120, 90])
    #expect(result.images.map(\.cameraIndex) == [0, 1])
}

@Test func buildResultFillsCameraParametersFromTheFrame() throws {
    // 90 x 120 has a 150 px diagonal, so f = 26 * 150 / 43.267.
    let frames = [photo("a.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png")],
        points: [],
        fallbackFocalLength35mm: nil)

    let camera = result.cameras[0]
    #expect(abs(camera.params[0] - 26.0 * 150.0 / 43.267) < 1e-9)
    #expect(camera.params[1] == 45)
    #expect(camera.params[2] == 60)
}

@Test func buildResultUsesFallbackFocalLengthWhenFrameHasNone() throws {
    let frames = [photo("a.png")]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png")],
        points: [],
        fallbackFocalLength35mm: 35)

    #expect(abs(result.cameras[0].params[0] - 35.0 * 150.0 / 43.267) < 1e-9)
}

@Test func buildResultFailsWhenNoFocalLengthIsKnown() {
    let frames = [photo("a.png")]
    #expect(throws: PoseEstimationFailure.failed(reason: "no focal length known for a.png")) {
        try ApplePhotogrammetryPoseProvider.buildResult(frames: frames,
                                                        poses: [pose("a.png")],
                                                        points: [],
                                                        fallbackFocalLength35mm: nil)
    }
}

@Test func buildResultDropsPosesWithUnknownNames() throws {
    let frames = [photo("a.png", focal35mm: 26), photo("b.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png"), pose("ghost.png"), pose("b.png")],
        points: [],
        fallbackFocalLength35mm: nil)

    #expect(result.images.map(\.name) == ["a.png", "b.png"])
}

@Test func buildResultSortsImagesByName() throws {
    let frames = [photo("a.png", focal35mm: 26),
                  photo("b.png", focal35mm: 26),
                  photo("c.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("c.png"), pose("a.png"), pose("b.png")],
        points: [],
        fallbackFocalLength35mm: nil)

    #expect(result.images.map(\.name) == ["a.png", "b.png", "c.png"])
}

@Test func buildResultConvertsPointsWithoutChangingPositions() throws {
    let frames = [photo("a.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png")],
        points: [(position: SIMD3<Double>(1, 2, 3), color: SIMD3<UInt8>(10, 20, 30))],
        fallbackFocalLength35mm: nil)

    #expect(result.points == [EstimatedPoint(position: SIMD3<Double>(1, 2, 3),
                                             color: SIMD3<UInt8>(10, 20, 30),
                                             error: 0)])
}

// MARK: - Apple intrinsics

@Test func intrinsicsWinOverConflictingFocalLength() throws {
    let frames = [photo("a.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png", intrinsics: validIntrinsics)],
        points: [],
        fallbackFocalLength35mm: nil)

    #expect(result.cameras.count == 1)
    #expect(result.cameras[0].modelID == 1)
    // PINHOLE parameter order: fx, fy, cx, cy.
    #expect(result.cameras[0].params == [1194.76, 1194.76, 45.2, 59.9])
    #expect(result.cameras[0].width == 90)
    #expect(result.cameras[0].height == 120)
    #expect(result.images[0].cameraIndex == 0)
}

@Test func identicalIntrinsicsShareOneCameraAndDifferentOnesDoNot() throws {
    let frames = [photo("a.png"), photo("b.png"), photo("c.png"), photo("d.png")]
    let other = SIMD4<Double>(1300, 1300, 45.2, 59.9)
    // Differs by 0.001 px, which rounds to the same 0.01 px value.
    let nearlyEqual = SIMD4<Double>(1194.761, 1194.761, 45.2, 59.9)
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png", intrinsics: validIntrinsics),
                pose("b.png", intrinsics: validIntrinsics),
                pose("c.png", intrinsics: other),
                pose("d.png", intrinsics: nearlyEqual)],
        points: [],
        fallbackFocalLength35mm: nil)

    #expect(result.cameras.count == 2)
    #expect(result.images.map(\.cameraIndex) == [0, 0, 1, 0])
}

@Test func invalidIntrinsicsFallBackToExifFocalLength() throws {
    let invalid: [SIMD4<Double>] = [
        SIMD4<Double>(0, 1194.76, 45.2, 59.9),           // zero fx
        SIMD4<Double>(1194.76, 0, 45.2, 59.9),           // zero fy
        SIMD4<Double>(1194.76, 1194.76, 0, 59.9),        // principal point on the left edge
        SIMD4<Double>(1194.76, 1194.76, 90, 59.9),       // cx equal to the width
        SIMD4<Double>(1194.76, 1194.76, 45.2, -1),       // cy negative
        SIMD4<Double>(1194.76, 1194.76, 45.2, 120),      // cy equal to the height
        SIMD4<Double>(Double.nan, 1194.76, 45.2, 59.9),  // NaN
        SIMD4<Double>(1194.76, Double.infinity, 45.2, 59.9),
    ]
    for values in invalid {
        let result = try ApplePhotogrammetryPoseProvider.buildResult(
            frames: [photo("a.png", focal35mm: 26)],
            poses: [pose("a.png", intrinsics: values)],
            points: [],
            fallbackFocalLength35mm: nil)
        #expect(result.cameras[0].modelID == 0)
        #expect(result.cameras[0].params.count == 3)
        #expect(abs(result.cameras[0].params[0] - 26.0 * 150.0 / 43.267) < 1e-9)
    }
}

@Test func invalidIntrinsicsWithoutFocalLengthFail() {
    let frames = [photo("a.png")]
    #expect(throws: PoseEstimationFailure.failed(reason: "no focal length known for a.png")) {
        try ApplePhotogrammetryPoseProvider.buildResult(
            frames: frames,
            poses: [pose("a.png", intrinsics: SIMD4<Double>(0, 1194.76, 45.2, 59.9))],
            points: [],
            fallbackFocalLength35mm: nil)
    }
}

@Test func mixedPosesUseIntrinsicsOrFocalLengthPerFrame() throws {
    let frames = [photo("a.png", focal35mm: 26),
                  photo("b.png", focal35mm: 26),
                  photo("c.png", focal35mm: 26)]
    let result = try ApplePhotogrammetryPoseProvider.buildResult(
        frames: frames,
        poses: [pose("a.png", intrinsics: validIntrinsics),
                pose("b.png"),
                pose("c.png", intrinsics: validIntrinsics)],
        points: [],
        fallbackFocalLength35mm: nil)

    // PINHOLE camera for a and c, SIMPLE_PINHOLE for b. The two models never share a camera.
    #expect(result.cameras.map(\.modelID) == [1, 0])
    #expect(result.images.map(\.cameraIndex) == [0, 1, 0])
}

// MARK: - Provider entry point

@Test func providerRejectsOneFrameBeforeTouchingRealityKit() async {
    let provider = ApplePhotogrammetryPoseProvider()
    #expect(provider.name == "Apple Photogrammetry")
    await #expect(throws: PoseEstimationFailure.tooFewFrames(have: 1, need: 2)) {
        try await provider.estimatePoses(for: [photo("a.png", focal35mm: 26)])
    }
}
#endif
