import Foundation

/// The three time/quality presets of the input pipeline (addendum decision A7). A preset sets image
/// size, iteration count and the schedules that depend on it together; individual values stay editable
/// afterwards. Regularizer strengths join the preset when the Phase 4 regularizers land.
public enum TrainingPreset: String, Codable, Sendable, CaseIterable, Equatable {
    case fast
    case standard
    case quality

    /// Preset used for new projects.
    public static let `default`: TrainingPreset = .standard

    /// Longer image side in pixels.
    public var maxImageSide: Int {
        switch self {
        case .fast: return 640
        case .standard: return 1_000
        case .quality: return 1_600
        }
    }

    public var iterations: Int {
        switch self {
        case .fast: return 3_000
        case .standard: return 10_000
        case .quality: return 15_000
        }
    }

    /// Writes the preset into `configuration`: image size, iterations and the schedules derived from
    /// them (densification window, scale pruning start, SH degree ramp). Other settings are kept.
    public func apply(to configuration: inout TrainingConfiguration) {
        let total: Int = iterations
        configuration.maxImageSide = maxImageSide
        configuration.iterations = total
        configuration.densification.startIteration = min(500, total / 10)
        configuration.densification.stopIteration = total * 6 / 10
        configuration.densification.pruneScaleStartIteration = total * 3 / 10
        configuration.shDegreeInterval = max(1, total / 10)
    }

    /// A configuration with the defaults and this preset applied.
    public func configuration() -> TrainingConfiguration {
        var configuration = TrainingConfiguration()
        apply(to: &configuration)
        return configuration
    }
}
