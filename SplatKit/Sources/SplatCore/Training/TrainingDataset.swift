import Foundation

/// Sparse point cloud used for Gaussian initialization.
public struct PointCloud: Sendable, Equatable {
    public var positions: [SIMD3<Float>]
    /// RGB in 0…1.
    public var colors: [SIMD3<Float>]

    public init(positions: [SIMD3<Float>], colors: [SIMD3<Float>]) {
        self.positions = positions
        self.colors = colors
    }
}

/// One training image with its camera.
public struct TrainingView: Sendable {
    public var name: String
    public var camera: Camera
    public var image: RGBImage
    /// Depth prior for this view. `nil` when no depth is available or it was rejected.
    public var depth: DepthPrior?

    public init(name: String, camera: Camera, image: RGBImage, depth: DepthPrior? = nil) {
        self.name = name
        self.camera = camera
        self.image = image
        self.depth = depth
    }
}

/// Views and optional initialization points for one training run.
public struct TrainingDataset: Sendable {
    public var views: [TrainingView]
    public var points: PointCloud?

    public init(views: [TrainingView], points: PointCloud? = nil) {
        self.views = views
        self.points = points
    }

    public var cameraCenters: [SIMD3<Float>] {
        views.map { $0.camera.center }
    }
}
