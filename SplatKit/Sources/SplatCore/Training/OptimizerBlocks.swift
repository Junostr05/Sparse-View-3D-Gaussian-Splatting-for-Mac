import Foundation

/// Adam state of the per-view colour compensation (v1.2) as stored in an optimizer snapshot. Plain data, so the
/// SplatMetal trainer state and the SplatIO serialisation share one type.
public struct AppearanceOptimizerBlock: Sendable, Equatable {
    /// Gain r, g, b then bias r, g, b per view (six values per view).
    public var parameters: [Float]
    public var firstMoment: [Float]
    public var secondMoment: [Float]
    /// Adam steps taken per view.
    public var stepCounts: [Int]

    public init(parameters: [Float], firstMoment: [Float], secondMoment: [Float], stepCounts: [Int]) {
        self.parameters = parameters
        self.firstMoment = firstMoment
        self.secondMoment = secondMoment
        self.stepCounts = stepCounts
    }

    /// Number of training views the block describes.
    public var viewCount: Int { stepCounts.count }
}

/// Camera pose refinement state (v1.2) as stored in an optimizer snapshot: the refined pose of each view and its
/// Adam state.
public struct PoseOptimizerBlock: Sendable, Equatable {
    /// Per view: rotation (column-major c0, c1, c2 as x, y, z triples, nine values) then translation (three values).
    public var poses: [Float]
    /// Six values per view: (ω, τ).
    public var firstMoment: [Double]
    public var secondMoment: [Double]
    /// Applied updates per view.
    public var steps: [Int]

    public init(poses: [Float], firstMoment: [Double], secondMoment: [Double], steps: [Int]) {
        self.poses = poses
        self.firstMoment = firstMoment
        self.secondMoment = secondMoment
        self.steps = steps
    }

    /// Number of training views the block describes.
    public var viewCount: Int { steps.count }
}
