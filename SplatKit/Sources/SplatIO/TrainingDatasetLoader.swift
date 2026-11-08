import Foundation
import SplatCore

/// A camera pose with the name of the image it belongs to. Carries no pixels.
public struct NamedCamera: Sendable, Equatable {
    public var name: String
    public var camera: Camera

    public init(name: String, camera: Camera) {
        self.name = name
        self.camera = camera
    }
}

/// Builds a `TrainingDataset` from COLMAP or ARKit input. Image bytes come from a caller
/// closure and are decoded by an injected decoder, so this module stays free of ImageIO.
public enum TrainingDatasetLoader {
    public typealias ImageDecoder = (Data) throws -> RGBImage

    /// Loads a COLMAP binary reconstruction. `points3D` is optional; without it the dataset has
    /// no initialization points. `imageData` returns the bytes of an image by its COLMAP name.
    public static func loadCOLMAP(cameras: Data,
                                  images: Data,
                                  points3D: Data?,
                                  imageData: (String) -> Data?,
                                  maxImageSide: Int,
                                  decode: ImageDecoder,
                                  excluding excluded: Set<String> = []) throws(DatasetError) -> TrainingDataset {
        let specs = try colmapSpecs(cameras: cameras, images: images).filter { !excluded.contains($0.name) }
        let views = try loadViews(specs,
                                  imageData: imageData,
                                  maxImageSide: maxImageSide,
                                  decode: decode)

        var points: PointCloud?
        if let points3D {
            points = pointCloud(try COLMAPReader.readPoints3D(points3D))
        }
        return TrainingDataset(views: views, points: points)
    }

    /// Loads an ARKit JSON capture (see `ARKitCapture`). ARKit provides no initialization points.
    public static func loadARKit(capture: Data,
                                 imageData: (String) -> Data?,
                                 maxImageSide: Int,
                                 decode: ImageDecoder,
                                 excluding excluded: Set<String> = []) throws(DatasetError) -> TrainingDataset {
        let specs = try arkitSpecs(capture: capture).filter { !excluded.contains($0.name) }
        let views = try loadViews(specs,
                                  imageData: imageData,
                                  maxImageSide: maxImageSide,
                                  decode: decode)
        return TrainingDataset(views: views, points: nil)
    }

    /// Camera poses of a COLMAP reconstruction without any image. Each camera has the pixel size
    /// from `cameras.bin`; distortion and image-size rescaling are not applied. Sorted by name.
    /// Returns an empty array when `colmapImages` lists no images.
    public static func cameras(colmapCameras: Data, colmapImages: Data) throws(DatasetError) -> [NamedCamera] {
        try namedCameras(try colmapSpecs(cameras: colmapCameras, images: colmapImages))
    }

    /// Camera poses of an ARKit JSON capture without any image. Same rules as
    /// `cameras(colmapCameras:colmapImages:)`.
    public static func cameras(arkitCapture: Data) throws(DatasetError) -> [NamedCamera] {
        try namedCameras(try arkitSpecs(capture: arkitCapture))
    }

    // MARK: - Shared view construction

    /// One view before its image is decoded: camera model, pixel size of the camera and pose.
    private struct ViewSpec {
        var name: String
        var model: CameraModel
        var width: Int
        var height: Int
        var rotation: Matrix3
        var translation: SIMD3<Float>
    }

    /// View specs of a COLMAP reconstruction, in image record order.
    private static func colmapSpecs(cameras: Data, images: Data) throws(DatasetError) -> [ViewSpec] {
        let cameraRecords = try COLMAPReader.readCameras(cameras)
        let imageRecords = try COLMAPReader.readImages(images)

        var cameraByID: [UInt32: COLMAPCamera] = [:]
        for record in cameraRecords {
            cameraByID[record.id] = record
        }

        var specs: [ViewSpec] = []
        specs.reserveCapacity(imageRecords.count)
        for image in imageRecords {
            guard let record = cameraByID[image.cameraID] else {
                throw .missingCamera(image: image.name, cameraID: image.cameraID)
            }
            let model = try CameraModel(colmap: record)
            let pose = try colmapPose(image)
            specs.append(ViewSpec(name: image.name,
                                  model: model,
                                  width: record.width,
                                  height: record.height,
                                  rotation: pose.rotation,
                                  translation: pose.translation))
        }
        return specs
    }

    /// View specs of an ARKit JSON capture, in frame order.
    private static func arkitSpecs(capture: Data) throws(DatasetError) -> [ViewSpec] {
        let parsed: ARKitCapture
        do {
            parsed = try JSONDecoder().decode(ARKitCapture.self, from: capture)
        } catch {
            throw .invalidARKitCapture(String(describing: error))
        }

        var specs: [ViewSpec] = []
        specs.reserveCapacity(parsed.frames.count)
        for frame in parsed.frames {
            let resolved = try ARKitCapture.resolve(frame)
            specs.append(ViewSpec(name: frame.image,
                                  model: resolved.model,
                                  width: frame.width,
                                  height: frame.height,
                                  rotation: resolved.rotation,
                                  translation: resolved.translation))
        }
        return specs
    }

    /// Cameras at their own pixel size, sorted by name in the same order `loadViews` uses. Does not
    /// read images, so an empty list is valid here (unlike `loadViews`, which throws `noViews`).
    private static func namedCameras(_ specs: [ViewSpec]) throws(DatasetError) -> [NamedCamera] {
        var result: [NamedCamera] = []
        result.reserveCapacity(specs.count)
        for spec in specs.sorted(by: { $0.name < $1.name }) {
            guard spec.width > 0, spec.height > 0 else {
                throw .truncated(file: "cameras.bin")
            }
            let camera = Camera(model: spec.model,
                                width: spec.width,
                                height: spec.height,
                                rotation: spec.rotation,
                                translation: spec.translation)
            result.append(NamedCamera(name: spec.name, camera: camera))
        }
        return result
    }

    /// Checks that every image exists (reporting all missing names at once), then decodes and
    /// prepares each view in name order.
    private static func loadViews(_ specs: [ViewSpec],
                                  imageData: (String) -> Data?,
                                  maxImageSide: Int,
                                  decode: ImageDecoder) throws(DatasetError) -> [TrainingView] {
        guard !specs.isEmpty else {
            throw .noViews
        }

        var entries: [(spec: ViewSpec, bytes: Data)] = []
        entries.reserveCapacity(specs.count)
        var missing: [String] = []
        for spec in specs {
            if let bytes = imageData(spec.name) {
                entries.append((spec, bytes))
            } else {
                missing.append(spec.name)
            }
        }
        guard missing.isEmpty else {
            throw .missingImages(missing.sorted())
        }

        entries.sort { $0.spec.name < $1.spec.name }
        var views: [TrainingView] = []
        views.reserveCapacity(entries.count)
        for entry in entries {
            views.append(try makeView(entry.spec,
                                      bytes: entry.bytes,
                                      maxImageSide: maxImageSide,
                                      decode: decode))
        }
        return views
    }

    /// Decodes one image, rescales intrinsics when the file is a different size, undistorts,
    /// and downsamples so the long side is at most `maxImageSide`.
    private static func makeView(_ spec: ViewSpec,
                                 bytes: Data,
                                 maxImageSide: Int,
                                 decode: ImageDecoder) throws(DatasetError) -> TrainingView {
        guard spec.width > 0, spec.height > 0 else {
            throw .truncated(file: "cameras.bin")
        }

        var image: RGBImage
        do {
            image = try decode(bytes)
        } catch {
            throw .imageDecodingFailed(name: spec.name, reason: String(describing: error))
        }
        guard image.width > 0, image.height > 0 else {
            throw .imageDecodingFailed(name: spec.name, reason: "image has no pixels")
        }

        var model = spec.model
        var width = spec.width
        var height = spec.height
        if image.width != spec.width || image.height != spec.height {
            let cameraAspect = Double(spec.width) / Double(spec.height)
            let imageAspect = Double(image.width) / Double(image.height)
            guard abs(imageAspect - cameraAspect) <= 0.01 * cameraAspect else {
                throw .imageSizeMismatch(name: spec.name,
                                         expected: SIMD2(spec.width, spec.height),
                                         actual: SIMD2(image.width, image.height))
            }
            model = model.rescaled(sx: Double(image.width) / Double(spec.width),
                                   sy: Double(image.height) / Double(spec.height))
            width = image.width
            height = image.height
        }

        if model.hasDistortion {
            image = Undistortion.undistort(image, model: model)
        }
        var camera = Camera(model: model,
                            width: width,
                            height: height,
                            rotation: spec.rotation,
                            translation: spec.translation)

        let factor = ImageResampling.downsampleFactor(width: width, height: height, maxSide: maxImageSide)
        if factor > 1 {
            image = image.downsampled(by: factor)
            camera = camera.downscaled(by: factor)
        }
        return TrainingView(name: spec.name, camera: camera, image: image)
    }

    // MARK: - COLMAP helpers

    /// COLMAP stores `(qw, qx, qy, qz)` world→camera; `Matrix3.rotation` takes `(w, x, y, z)`.
    private static func colmapPose(_ image: COLMAPImage) throws(DatasetError) -> (rotation: Matrix3, translation: SIMD3<Float>) {
        let q = image.rotation
        let t = image.translation
        let finite = q.x.isFinite && q.y.isFinite && q.z.isFinite && q.w.isFinite
            && t.x.isFinite && t.y.isFinite && t.z.isFinite
        let lengthSquared = (q * q).sum()
        guard finite, lengthSquared > 0 else {
            throw .invalidPose(name: image.name)
        }
        let rotation = Matrix3.rotation(quaternion: SIMD4(Float(q.x), Float(q.y), Float(q.z), Float(q.w)))
        let translation = SIMD3(Float(t.x), Float(t.y), Float(t.z))
        return (rotation, translation)
    }

    private static func pointCloud(_ points: [COLMAPPoint]) -> PointCloud {
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        positions.reserveCapacity(points.count)
        colors.reserveCapacity(points.count)
        for point in points {
            positions.append(SIMD3(Float(point.position.x), Float(point.position.y), Float(point.position.z)))
            colors.append(SIMD3(Float(point.color.x) / 255,
                                Float(point.color.y) / 255,
                                Float(point.color.z) / 255))
        }
        return PointCloud(positions: positions, colors: colors)
    }
}

// MARK: - Intrinsics rescaling

extension CameraModel {
    /// Same model for an image resized by `sx` horizontally and `sy` vertically. Focal lengths
    /// and principal point scale with the image; distortion coefficients act on normalized
    /// coordinates and stay unchanged. Single-focal models use the mean of `sx` and `sy`.
    fileprivate func rescaled(sx: Double, sy: Double) -> CameraModel {
        let meanScale = (sx + sy) / 2
        switch self {
        case let .simplePinhole(f, cx, cy):
            return .simplePinhole(f: f * meanScale, cx: cx * sx, cy: cy * sy)
        case let .pinhole(fx, fy, cx, cy):
            return .pinhole(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy)
        case let .simpleRadial(f, cx, cy, k):
            return .simpleRadial(f: f * meanScale, cx: cx * sx, cy: cy * sy, k: k)
        case let .radial(f, cx, cy, k1, k2):
            return .radial(f: f * meanScale, cx: cx * sx, cy: cy * sy, k1: k1, k2: k2)
        case let .opencv(fx, fy, cx, cy, k1, k2, p1, p2):
            return .opencv(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy,
                           k1: k1, k2: k2, p1: p1, p2: p2)
        }
    }
}
