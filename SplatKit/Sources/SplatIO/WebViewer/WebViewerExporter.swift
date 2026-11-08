import Foundation
import SplatCore

/// The camera a web page starts from, in the cloud's own coordinates (the frame the viewer renders in).
public struct WebViewerCamera: Sendable, Equatable {
    /// Eye position.
    public var position: SIMD3<Float>
    /// Point the camera looks at. Orbiting turns the view around this point.
    public var target: SIMD3<Float>
    /// World up direction. The page orbits around it.
    public var up: SIMD3<Float>
    /// Vertical field of view in radians.
    public var fovY: Float

    public init(position: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>, fovY: Float) {
        self.position = position
        self.target = target
        self.up = up
        self.fovY = fovY
    }
}

/// Settings for `WebViewerExporter`.
public struct WebViewerOptions: Sendable, Equatable {
    /// Default cap on the number of Gaussians embedded in the page.
    public static let defaultMaxGaussians: Int = 1_000_000

    /// Largest number of Gaussians to embed. A larger cloud keeps the most important ones. Values below 1 count as 1.
    public var maxGaussians: Int
    /// Page title, shown in the browser tab and in the overlay.
    public var title: String
    /// Initial camera. `nil` frames the whole cloud along `upAxis`, as the viewer does for a new scene.
    public var camera: WebViewerCamera?
    /// Background colour, sRGB components in 0...1.
    public var background: SIMD3<Float>
    /// Up axis used when `camera` is `nil`. The page uses the camera's own up vector otherwise.
    public var upAxis: UpAxis

    public init(maxGaussians: Int = WebViewerOptions.defaultMaxGaussians,
                title: String = "Gaussian splat",
                camera: WebViewerCamera? = nil,
                background: SIMD3<Float> = .zero,
                upAxis: UpAxis = .negativeY) {
        self.maxGaussians = maxGaussians
        self.title = title
        self.camera = camera
        self.background = background
        self.upAxis = upAxis
    }
}

/// The result of `WebViewerExporter.export`.
public struct WebViewerExport: Sendable, Equatable {
    /// The UTF-8 HTML document.
    public let data: Data
    /// Gaussians embedded in the page.
    public let includedCount: Int
    /// Gaussians left out because the cloud exceeded `maxGaussians`. These are the least important ones.
    public let droppedCount: Int
    /// Gaussians with a NaN or infinite value, removed before anything else.
    public let removedNonFiniteCount: Int

    public init(data: Data, includedCount: Int, droppedCount: Int, removedNonFiniteCount: Int) {
        self.data = data
        self.includedCount = includedCount
        self.droppedCount = droppedCount
        self.removedNonFiniteCount = removedNonFiniteCount
    }
}

/// Writes a single self-contained HTML file that shows a cloud in any WebGL 2 browser.
///
/// The Gaussians are embedded as `.splat` records (32 bytes each, base colour only), gzip-compressed and
/// base64-encoded in a `<script type="application/octet-stream" id="data">` block. The page decodes them with
/// `DecompressionStream('gzip')` and renders them with the EWA splatting described in `WebViewerPage`.
/// When the cloud is at most 8 MB raw, an uncompressed copy is also embedded for browsers without
/// `DecompressionStream`.
///
/// The file contains no external references: no scripts, stylesheets, fonts or requests.
///
/// Size: about `32 × count × 0.6 × 4/3` bytes, see `estimatedByteCount(gaussianCount:)`.
public enum WebViewerExporter {
    /// Bytes per Gaussian in the embedded records, the `.splat` layout of `SplatWriter`.
    static let recordSize: Int = 32
    /// Largest raw record payload for which an uncompressed fallback block is also embedded.
    static let rawFallbackLimit: Int = 8 * 1024 * 1024

    /// Rough size of the finished page for `gaussianCount` Gaussians (32 bytes each, about 0.6 after gzip,
    /// 4/3 for base64). The real size depends on the data.
    public static func estimatedByteCount(gaussianCount: Int) -> Int {
        let raw: Double = Double(max(0, gaussianCount)) * Double(recordSize)
        let estimate: Double = raw * 0.6 * 4.0 / 3.0
        return Int(estimate.rounded())
    }

    /// Builds the page. Throws `.empty` when no Gaussian is left after the non-finite values are removed.
    public static func export(cloud: GaussianCloud,
                              options: WebViewerOptions = WebViewerOptions()) throws(SplatIOError) -> WebViewerExport {
        let cleaned: (cloud: GaussianCloud, removed: Int) = cloud.removingNonFinite()
        let finite: GaussianCloud = cleaned.cloud
        guard finite.count > 0 else {
            throw .empty
        }
        let limit: Int = max(1, options.maxGaussians)
        let included: Int = min(finite.count, limit)
        let dropped: Int = finite.count - included

        // SplatWriter orders records by descending importance (exp(sum of log scales) × opacity), ties by
        // index. The first `included` records are therefore the most important Gaussians.
        let sorted: Data = SplatWriter.write(finite)
        let records: [UInt8] = [UInt8](sorted.prefix(included * recordSize))

        let camera: WebViewerCamera = resolvedCamera(options.camera, cloud: finite, upAxis: options.upAxis)
        let config: String = try configJSON(title: options.title,
                                            count: included,
                                            total: finite.count,
                                            camera: camera,
                                            upAxis: options.upAxis,
                                            background: options.background)

        let dataBase64: String = Data(Gzip.compress(records)).base64EncodedString()
        var rawBlock: String = ""
        if records.count < rawFallbackLimit {
            let rawBase64: String = Data(records).base64EncodedString()
            rawBlock = "<script id=\"data-raw\" type=\"application/octet-stream\">\(rawBase64)</script>"
        }

        let html: String = WebViewerPage.render(title: htmlEscaped(options.title),
                                                config: config,
                                                dataBase64: dataBase64,
                                                rawBlock: rawBlock)
        return WebViewerExport(data: Data(html.utf8),
                               includedCount: included,
                               droppedCount: dropped,
                               removedNonFiniteCount: cleaned.removed)
    }

    // MARK: - Camera

    /// The requested camera when it is usable, otherwise the viewer's default framing of the cloud.
    static func resolvedCamera(_ requested: WebViewerCamera?, cloud: GaussianCloud, upAxis: UpAxis) -> WebViewerCamera {
        if let requested, isUsable(requested) {
            return requested
        }
        let bounds: Bounds = boundsOf(cloud)
        let view: ViewCamera = ViewCamera.framing(bounds, upAxis: upAxis)
        let framed = WebViewerCamera(position: view.eye, target: view.pivot, up: upAxis.vector, fovY: view.fovY)
        if isUsable(framed) {
            return framed
        }
        // Degenerate bounds: look at the centre from one unit away, along a direction perpendicular to up.
        let up: SIMD3<Float> = upAxis.vector
        let center: SIMD3<Float> = bounds.min + (bounds.max - bounds.min) * 0.5
        let side: SIMD3<Float> = abs(up.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 0, 1)
        return WebViewerCamera(position: center + side, target: center, up: up, fovY: OrbitCamera.defaultFovY)
    }

    static func isUsable(_ camera: WebViewerCamera) -> Bool {
        let values: [SIMD3<Float>] = [camera.position, camera.target, camera.up]
        for vector in values {
            for c in 0..<3 where !vector[c].isFinite {
                return false
            }
        }
        let offset: SIMD3<Float> = camera.target - camera.position
        guard (offset * offset).sum() > 1e-12 else { return false }
        guard (camera.up * camera.up).sum() > 1e-12 else { return false }
        guard camera.fovY.isFinite, camera.fovY > 0.01, camera.fovY < Float.pi - 0.01 else { return false }
        return true
    }

    private static func boundsOf(_ cloud: GaussianCloud) -> Bounds {
        var lower = SIMD3<Float>(repeating: Float.infinity)
        var upper = SIMD3<Float>(repeating: -Float.infinity)
        for i in 0..<cloud.count {
            for c in 0..<3 {
                let value: Float = cloud.positions[3 * i + c]
                lower[c] = Swift.min(lower[c], value)
                upper[c] = Swift.max(upper[c], value)
            }
        }
        return Bounds(min: lower, max: upper)
    }

    // MARK: - Configuration

    private struct CameraConfig: Encodable {
        let position: [Double]
        let target: [Double]
        /// Vertical field of view in radians.
        let fovY: Double
    }

    private struct PageConfig: Encodable {
        let title: String
        /// Gaussians embedded in the page.
        let count: Int
        /// Gaussians in the cloud after non-finite values were removed.
        let total: Int
        let upAxis: String
        let up: [Double]
        let camera: CameraConfig
        let background: [Double]
    }

    /// The configuration JSON, made safe to embed in a script element. Every `<`, `>`, `&` and `%` inside the
    /// JSON becomes a `\u` escape. `%` is escaped so that the template placeholders cannot appear in the title.
    static func configJSON(title: String,
                           count: Int,
                           total: Int,
                           camera: WebViewerCamera,
                           upAxis: UpAxis,
                           background: SIMD3<Float>) throws(SplatIOError) -> String {
        let config = PageConfig(
            title: title,
            count: count,
            total: total,
            upAxis: upAxis.rawValue,
            up: vectorValues(camera.up),
            camera: CameraConfig(position: vectorValues(camera.position),
                                 target: vectorValues(camera.target),
                                 fovY: Double(camera.fovY)),
            background: vectorValues(background)
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let encoded = try? encoder.encode(config) else {
            throw .nonFinite
        }
        let json: String = String(decoding: encoded, as: UTF8.self)
        return json
            .replacingOccurrences(of: "<", with: "\\u003c")
            .replacingOccurrences(of: ">", with: "\\u003e")
            .replacingOccurrences(of: "&", with: "\\u0026")
            .replacingOccurrences(of: "%", with: "\\u0025")
    }

    private static func vectorValues(_ vector: SIMD3<Float>) -> [Double] {
        [Double(vector.x), Double(vector.y), Double(vector.z)]
    }

    /// Escapes text for an HTML element body.
    static func htmlEscaped(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
    }
}
