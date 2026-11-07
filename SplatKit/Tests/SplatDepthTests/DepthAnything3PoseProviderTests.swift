import CoreGraphics
import Foundation
import ImageIO
import SplatCore
@testable import SplatDepth
import SplatIO
import Testing

/// Converted Depth Anything 3 package for the opt-in Core ML smoke test. Unset means the test is skipped.
private let depthAnything3PackageURL: URL? = ProcessInfo.processInfo.environment["DA3_PACKAGE"]
    .map { (path: String) -> URL in URL(fileURLWithPath: path, isDirectory: true) }

/// A portrait PNG with a colour gradient, shifted by `shift` so that the frames differ from one another.
private func makeGradientPNG(width: Int, height: Int, shift: Int) throws -> Data {
    let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(data: nil,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width * 4,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
          let base = context.data else {
        throw CocoaError(.fileWriteUnknown)
    }
    let pixels: UnsafeMutableRawPointer = base
    for row in 0..<height {
        for column in 0..<width {
            let offset: Int = (row * width + column) * 4
            let red: UInt8 = UInt8((column * 255) / (width - 1))
            let green: UInt8 = UInt8((row * 255) / (height - 1))
            let blue: UInt8 = UInt8((((column + row + shift * 37) % 256) + 256) % 256)
            pixels.storeBytes(of: red, toByteOffset: offset, as: UInt8.self)
            pixels.storeBytes(of: green, toByteOffset: offset + 1, as: UInt8.self)
            pixels.storeBytes(of: blue, toByteOffset: offset + 2, as: UInt8.self)
            pixels.storeBytes(of: UInt8(255), toByteOffset: offset + 3, as: UInt8.self)
        }
    }
    guard let image = context.makeImage() else {
        throw CocoaError(.fileWriteUnknown)
    }
    let encoded: NSMutableData = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(encoded, "public.png" as CFString, 1, nil) else {
        throw CocoaError(.fileWriteUnknown)
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw CocoaError(.fileWriteUnknown)
    }
    return encoded as Data
}

private func syntheticFrames(count: Int, width: Int, height: Int) throws -> [CapturedFrame] {
    var frames: [CapturedFrame] = []
    for index in 0..<count {
        let name: String = String(format: "frame_%04d.png", index)
        let data: Data = try makeGradientPNG(width: width, height: height, shift: index)
        frames.append(CapturedFrame(name: name, data: data, sourceName: name, timestamp: nil))
    }
    return frames
}

@Suite struct DepthAnything3PoseProviderTests {
    @Test func tooFewFramesIsReportedBeforeAnyModelWork() async throws {
        let provider = DepthAnything3PoseProvider(packageURL: URL(fileURLWithPath: "/nonexistent/DA3.mlpackage", isDirectory: true))
        let frames = try syntheticFrames(count: 1, width: 40, height: 60)
        do {
            _ = try await provider.estimatePoses(for: frames)
            Issue.record("one frame must not produce poses")
        } catch {
            #expect(error == .tooFewFrames(have: 1, need: 2))
        }
    }

    @Test func missingPackageIsUnavailable() async throws {
        let missing = URL(fileURLWithPath: "/nonexistent/DepthAnything3BaseQ8.mlpackage", isDirectory: true)
        let provider = DepthAnything3PoseProvider(packageURL: missing)
        let frames = try syntheticFrames(count: 6, width: 45, height: 60)
        do {
            _ = try await provider.estimatePoses(for: frames)
            Issue.record("a missing package must not produce poses")
        } catch {
            guard case .unavailable = error else {
                Issue.record("expected .unavailable, got \(error)")
                return
            }
        }
    }

    @Test(.enabled(if: depthAnything3PackageURL != nil, "set DA3_PACKAGE to a converted Depth Anything 3 package"))
    func smokeRunOnSixPortraitFrames() async throws {
        let packageURL: URL = try #require(depthAnything3PackageURL)
        let provider = DepthAnything3PoseProvider(packageURL: packageURL)
        let frames = try syntheticFrames(count: 6, width: 450, height: 600)

        let clock = ContinuousClock()
        let start = clock.now
        let result: PoseEstimationResult
        do {
            result = try await provider.estimatePoses(for: frames)
        } catch {
            Issue.record("the smoke run failed: \(error)")
            return
        }
        let elapsed = start.duration(to: clock.now)
        print("DA3 smoke test: \(elapsed) for \(frames.count) frames, \(result.images.count) poses, \(result.points.count) points")

        #expect(result.images.count == 6)
        #expect(result.cameras.count == 6)
        #expect(result.registeredImageNames == frames.map(\.name).sorted())
        for image in result.images {
            let rotation: SIMD4<Double> = image.rotation
            let norm: Double = (rotation[0] * rotation[0] + rotation[1] * rotation[1]
                                + rotation[2] * rotation[2] + rotation[3] * rotation[3]).squareRoot()
            #expect(abs(norm - 1) < 1e-6, "rotation of \(image.name) is not a unit quaternion")
            #expect(rotation.x.isFinite && rotation.y.isFinite && rotation.z.isFinite && rotation.w.isFinite)
            #expect(image.translation.x.isFinite && image.translation.y.isFinite && image.translation.z.isFinite)
        }
        for camera in result.cameras {
            #expect(camera.modelID == 1, "Depth Anything 3 cameras are PINHOLE")
            #expect(camera.params.allSatisfy { (value: Double) -> Bool in value.isFinite && value > 0 })
        }
        #expect(result.points.count > 100)
        #expect(result.points.count <= 20_000)
        #expect(result.points.allSatisfy { (point: EstimatedPoint) -> Bool in
            point.position.x.isFinite && point.position.y.isFinite && point.position.z.isFinite
        })
    }
}
