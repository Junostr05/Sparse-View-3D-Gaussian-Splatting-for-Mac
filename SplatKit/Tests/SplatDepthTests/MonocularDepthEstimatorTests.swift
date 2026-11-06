import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import SplatCore
@testable import SplatDepth
import SplatIO
import Testing

// MARK: - Helpers

/// A small RGB image whose pixel (x, y) is (x / w, y / h, 0.25), so every position is distinguishable.
private func gradientImage(width: Int, height: Int) -> RGBImage {
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for row in 0..<height {
        for column in 0..<width {
            let red: Float = Float(column) / Float(width)
            let green: Float = Float(row) / Float(height)
            let pixel: SIMD3<Float> = SIMD3<Float>(red, green, 0.25)
            pixels.append(pixel)
        }
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

private func byteValue(_ value: Float) -> Int {
    let scaled: Float = min(max(value, 0), 1) * 255
    return Int(scaled.rounded())
}

/// Reads one BGRA pixel from a locked buffer and returns its (R, G, B) bytes.
private func bgraPixel(_ buffer: CVPixelBuffer, column: Int, row: Int) -> (red: Int, green: Int, blue: Int) {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let base: UnsafeRawPointer = UnsafeRawPointer(CVPixelBufferGetBaseAddress(buffer)!)
    let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(buffer)
    let offset: Int = row * bytesPerRow + column * 4
    let blue: UInt8 = base.load(fromByteOffset: offset, as: UInt8.self)
    let green: UInt8 = base.load(fromByteOffset: offset + 1, as: UInt8.self)
    let red: UInt8 = base.load(fromByteOffset: offset + 2, as: UInt8.self)
    return (Int(red), Int(green), Int(blue))
}

/// Builds a one-component buffer of the given format from row-major Float values.
private func makeOutputBuffer(width: Int, height: Int, format: OSType, values: [Float]) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, nil, &buffer)
    precondition(status == kCVReturnSuccess && buffer != nil, "test buffer allocation failed")
    let pixelBuffer: CVPixelBuffer = buffer!
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    let base: UnsafeMutableRawPointer = CVPixelBufferGetBaseAddress(pixelBuffer)!
    let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
    for row in 0..<height {
        for column in 0..<width {
            let value: Float = values[row * width + column]
            if format == kCVPixelFormatType_OneComponent16Half {
                let bits: UInt16 = Float16(value).bitPattern
                base.storeBytes(of: bits, toByteOffset: row * bytesPerRow + column * 2, as: UInt16.self)
            } else {
                base.storeBytes(of: value, toByteOffset: row * bytesPerRow + column * 4, as: Float.self)
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
    return pixelBuffer
}

/// Decodes an image file into an sRGB `RGBImage` with values in 0...1.
private func loadRGBImage(at url: URL) throws -> RGBImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let width: Int = image.width
    let height: Int = image.height
    let colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bytesPerRow: Int = width * 4
    var bytes: [UInt8] = [UInt8](repeating: 0, count: bytesPerRow * height)
    bytes.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
        let context: CGContext = CGContext(data: raw.baseAddress, width: width, height: height,
                                           bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace,
                                           bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for index in 0..<(width * height) {
        let red: Float = Float(bytes[index * 4]) / 255
        let green: Float = Float(bytes[index * 4 + 1]) / 255
        let blue: Float = Float(bytes[index * 4 + 2]) / 255
        pixels.append(SIMD3<Float>(red, green, blue))
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

private func meanValue(_ map: DepthMap, columns: Range<Int>, rows: Range<Int>) -> Double {
    var total: Double = 0
    var count: Int = 0
    for row in rows {
        for column in columns {
            total += Double(map.values[row * map.width + column])
            count += 1
        }
    }
    return total / Double(count)
}

private func temporaryDirectory(_ name: String) throws -> URL {
    let url: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent("MonocularDepthEstimatorTests-\(name)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

// MARK: - Model-free tests

@Test func absentModelReturnsNil() throws {
    let root: URL = try temporaryDirectory("absent")
    defer { try? FileManager.default.removeItem(at: root) }
    let store: ModelStore = ModelStore(rootDirectory: root)
    let estimator: MonocularDepthEstimator? = try MonocularDepthEstimator.installed(
        in: store, descriptor: ModelCatalog.depthAnythingV2Small)
    #expect(estimator == nil)
}

@Test func portraitRotationMovesTopLeftToTopRight() {
    // 2 wide, 3 high. Clockwise: the top-left pixel moves to the top-right corner of a 3-wide, 2-high image.
    let image: RGBImage = gradientImage(width: 2, height: 3)
    let rotated: RGBImage = MonocularDepthImageSupport.rotatedClockwise(image)
    #expect(rotated.width == 3)
    #expect(rotated.height == 2)
    #expect(rotated.pixels[2] == image.pixels[0])
    #expect(rotated.pixels[0] == image.pixels[4])
}

@Test func modelOutputRotatedBackKeepsViewOrientation() {
    // The estimator rotates the model output counter-clockwise: a 518-wide, 392-high output of a
    // rotated portrait view becomes a 392-wide, 518-high map in the view's own orientation.
    let output: DepthMap = DepthMap(width: 5, height: 4, kind: .relativeInverseDepth,
                                    values: (0..<20).map { Float($0 + 1) }, isEstimated: true)
    let restored: DepthMap = output.rotatedCounterClockwise()
    #expect(restored.width == 4)
    #expect(restored.height == 5)
    #expect(restored.isEstimated)
    #expect(restored.kind == .relativeInverseDepth)
    #expect(restored.rotatedClockwise() == output)
}

@Test func bufferConversionKeepsPixelValuesAtInputSize() throws {
    // 2x2 image with exact colours, converted at its own size: each byte is the rounded 8-bit value.
    let pixels: [SIMD3<Float>] = [
        SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(0, 0, 1), SIMD3<Float>(0.5, 0.5, 0.5),
    ]
    let image: RGBImage = RGBImage(width: 2, height: 2, pixels: pixels)
    let buffer: CVPixelBuffer = try MonocularDepthImageSupport.inputBuffer(from: image, width: 2, height: 2)
    #expect(CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA)
    #expect(CVPixelBufferGetWidth(buffer) == 2)
    #expect(CVPixelBufferGetHeight(buffer) == 2)
    let red: (red: Int, green: Int, blue: Int) = bgraPixel(buffer, column: 0, row: 0)
    #expect(red.red == 255 && red.green == 0 && red.blue == 0)
    let green: (red: Int, green: Int, blue: Int) = bgraPixel(buffer, column: 1, row: 0)
    #expect(green.red == 0 && green.green == 255 && green.blue == 0)
    let blue: (red: Int, green: Int, blue: Int) = bgraPixel(buffer, column: 0, row: 1)
    #expect(blue.red == 0 && blue.green == 0 && blue.blue == 255)
    let grey: (red: Int, green: Int, blue: Int) = bgraPixel(buffer, column: 1, row: 1)
    #expect(abs(grey.red - 128) <= 1 && abs(grey.green - 128) <= 1 && abs(grey.blue - 128) <= 1)
}

@Test func bufferConversionStretchesToModelSize() throws {
    // A uniform colour stays uniform when stretched from 2x2 to 5x3 (no letterboxing).
    let colour: SIMD3<Float> = SIMD3<Float>(0.2, 0.6, 0.9)
    let image: RGBImage = RGBImage(width: 2, height: 2, pixels: [SIMD3<Float>](repeating: colour, count: 4))
    let buffer: CVPixelBuffer = try MonocularDepthImageSupport.inputBuffer(from: image, width: 5, height: 3)
    #expect(CVPixelBufferGetWidth(buffer) == 5)
    #expect(CVPixelBufferGetHeight(buffer) == 3)
    for row in 0..<3 {
        for column in 0..<5 {
            let pixel: (red: Int, green: Int, blue: Int) = bgraPixel(buffer, column: column, row: row)
            #expect(abs(pixel.red - byteValue(colour.x)) <= 1)
            #expect(abs(pixel.green - byteValue(colour.y)) <= 1)
            #expect(abs(pixel.blue - byteValue(colour.z)) <= 1)
        }
    }
}

@Test func outputHalfFloatMapsInvalidValuesToZero() throws {
    let values: [Float] = [0.5, 0, -1, Float.nan, 2, Float.infinity]
    let buffer: CVPixelBuffer = makeOutputBuffer(width: 3, height: 2, format: kCVPixelFormatType_OneComponent16Half,
                                                 values: values)
    let map: DepthMap = try MonocularDepthImageSupport.depthMap(from: buffer)
    #expect(map.width == 3)
    #expect(map.height == 2)
    #expect(map.kind == .relativeInverseDepth)
    #expect(map.isEstimated)
    #expect(map.confidence == nil)
    #expect(map.values == [0.5, 0, 0, 0, 2, 0])
}

@Test func outputFloatMapKeepsPositiveValues() throws {
    let values: [Float] = [0.25, -3, Float.nan, 4]
    let buffer: CVPixelBuffer = makeOutputBuffer(width: 2, height: 2, format: kCVPixelFormatType_OneComponent32Float,
                                                 values: values)
    let map: DepthMap = try MonocularDepthImageSupport.depthMap(from: buffer)
    #expect(map.values == [0.25, 0, 0, 4])
}

@Test func outputWithUnsupportedFormatIsRejected() {
    let buffer: CVPixelBuffer = makeOutputBuffer(width: 2, height: 2, format: kCVPixelFormatType_32BGRA,
                                                 values: [0, 0, 0, 0])
    #expect(throws: MonocularDepthError.self) {
        _ = try MonocularDepthImageSupport.depthMap(from: buffer)
    }
}

// MARK: - Model-dependent tests (skipped unless SPLAT_DEPTH_MODEL names a .mlpackage)

/// The `.mlpackage` named by `SPLAT_DEPTH_MODEL`, or nil when the variable is unset.
private func configuredModelURL() -> URL? {
    guard let path = ProcessInfo.processInfo.environment["SPLAT_DEPTH_MODEL"], !path.isEmpty else {
        return nil
    }
    return URL(fileURLWithPath: path)
}

/// Copies the package into a fresh temporary directory, so Core ML can cache its compiled model
/// next to the copy and the configured package stays untouched.
private func copyOfModel(_ source: URL) throws -> (directory: URL, package: URL) {
    let directory: URL = try temporaryDirectory("model")
    let package: URL = directory.appendingPathComponent(source.lastPathComponent, isDirectory: true)
    try FileManager.default.copyItem(at: source, to: package)
    return (directory, package)
}

extension Camera {
    /// A camera with an identity pose and a centred principal point. Only the image size matters to the estimator.
    static func placeholder(width: Int, height: Int) -> Camera {
        Camera(width: width, height: height, fx: 60, fy: 60, cx: Float(width) / 2, cy: Float(height) / 2,
               rotation: Matrix3.identity, translation: SIMD3<Float>(0, 0, 0))
    }
}

/// The 900x1200 portrait test photo. `SPLAT_DEPTH_TEST_PHOTO` overrides the default location, which is
/// `sets/png13/IMG_9018.png` in the scratchpad that holds the model copy (four levels above the package).
private func photograph() throws -> RGBImage {
    let url: URL
    if let override = ProcessInfo.processInfo.environment["SPLAT_DEPTH_TEST_PHOTO"], !override.isEmpty {
        url = URL(fileURLWithPath: override)
    } else {
        let model: URL = try #require(configuredModelURL())
        url = model
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("sets/png13/IMG_9018.png")
    }
    return try loadRGBImage(at: url)
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["SPLAT_DEPTH_MODEL"] != nil,
               "Set SPLAT_DEPTH_MODEL to a DepthAnythingV2SmallF16.mlpackage to run the model tests"))
func predictsPortraitAndLandscapeAtViewOrientation() throws {
    let source: URL = try #require(configuredModelURL())
    let copy = try copyOfModel(source)
    defer { try? FileManager.default.removeItem(at: copy.directory) }
    let estimator: MonocularDepthEstimator = try MonocularDepthEstimator(packageURL: copy.package)

    let portrait: RGBImage = try photograph()
    #expect(portrait.height > portrait.width)
    let portraitView: TrainingView = TrainingView(name: "portrait", camera: Camera.placeholder(width: portrait.width,
                                                                                               height: portrait.height),
                                                  image: portrait)
    let portraitMap: DepthMap = try #require(try estimator.depthMap(for: portraitView))
    #expect(portraitMap.width == 392)
    #expect(portraitMap.height == 518)

    let landscapeView: TrainingView = TrainingView(name: "landscape",
                                                   camera: Camera.placeholder(width: portrait.height,
                                                                              height: portrait.width),
                                                   image: MonocularDepthImageSupport.rotatedClockwise(portrait))
    let landscapeMap: DepthMap = try #require(try estimator.depthMap(for: landscapeView))
    #expect(landscapeMap.width == 518)
    #expect(landscapeMap.height == 392)
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["SPLAT_DEPTH_MODEL"] != nil,
               "Set SPLAT_DEPTH_MODEL to a DepthAnythingV2SmallF16.mlpackage to run the model tests"))
func photographDepthIsFiniteAndNearIsLarger() throws {
    let source: URL = try #require(configuredModelURL())
    let copy = try copyOfModel(source)
    defer { try? FileManager.default.removeItem(at: copy.directory) }

    let coldStart: Date = Date()
    let estimator: MonocularDepthEstimator = try MonocularDepthEstimator(packageURL: copy.package)
    let photo: RGBImage = try photograph()
    let view: TrainingView = TrainingView(name: "IMG_9018",
                                          camera: Camera.placeholder(width: photo.width, height: photo.height),
                                          image: photo)
    let map: DepthMap = try #require(try estimator.depthMap(for: view))
    print("MonocularDepthEstimator cold (compile + first predict): \(Date().timeIntervalSince(coldStart)) s")

    let warmStart: Date = Date()
    let warm: DepthMap = try #require(try estimator.depthMap(for: view))
    print("MonocularDepthEstimator warm predict: \(Date().timeIntervalSince(warmStart)) s")
    #expect(warm == map)

    // Cached compiled model: a second estimator loads without compiling again.
    let reloadStart: Date = Date()
    _ = try MonocularDepthEstimator(packageURL: copy.package)
    print("MonocularDepthEstimator reload with cache: \(Date().timeIntervalSince(reloadStart)) s")

    #expect(map.width == 392)
    #expect(map.height == 518)
    #expect(map.values.count == 392 * 518)
    #expect(map.values.allSatisfy { $0.isFinite && $0 > 0 })
    let minimum: Float = map.values.min() ?? 0
    let maximum: Float = map.values.max() ?? 0
    let mean: Double = map.values.reduce(0.0) { $0 + Double($1) } / Double(map.values.count)
    print("IMG_9018 output min \(minimum) max \(maximum) mean \(mean)")
    #expect(minimum > 0)

    // Near: the coffee table objects (CD wallets and the red mug). Far: the floor strip at the back,
    // between the cabinet and the glass table, on the left of the photo. Both regions are in the
    // 900x1200 view coordinates and are scaled to the 392x518 map.
    let scaleX: Double = Double(map.width) / Double(photo.width)
    let scaleY: Double = Double(map.height) / Double(photo.height)
    let nearColumns: Range<Int> = Int(260 * scaleX)..<Int(540 * scaleX)
    let nearRows: Range<Int> = Int(520 * scaleY)..<Int(760 * scaleY)
    let farColumns: Range<Int> = Int(10 * scaleX)..<Int(160 * scaleX)
    let farRows: Range<Int> = Int(462 * scaleY)..<Int(478 * scaleY)
    let nearMean: Double = meanValue(map, columns: nearColumns, rows: nearRows)
    let farMean: Double = meanValue(map, columns: farColumns, rows: farRows)
    print("IMG_9018 near region mean \(nearMean), far floor mean \(farMean)")
    #expect(nearMean > farMean)
}
