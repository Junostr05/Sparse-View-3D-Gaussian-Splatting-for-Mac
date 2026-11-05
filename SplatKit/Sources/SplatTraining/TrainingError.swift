import SplatMetal

/// Errors of the Metal training loop (`GPUTrainer`, `TrainingSession`).
///
/// `metal` carries a message rather than the `MetalError` itself so that the type stays `Equatable`
/// and can be sent through `TrainingProgress.State.failed` without extra conversion.
public enum TrainingError: Error, Equatable {
    /// A Metal call failed (shader, buffer, command buffer or GPU error). The message is human readable.
    case metal(String)
    /// The dataset has no training views.
    case noViews
    /// The loss of the step with 0-based index `iteration` was NaN or infinite.
    case nonFiniteLoss(iteration: Int)
    /// The initial cloud has more Gaussians than the memory budget allows (`cap`).
    case initialCloudExceedsBudget(count: Int, cap: Int)
    /// `run()` was already called on this session (a session runs once), or a run is in progress.
    case alreadyRunning
}

/// Runs a Metal call and converts its `MetalError` into `TrainingError.metal`.
func metalCall<T>(_ body: () throws(MetalError) -> T) throws(TrainingError) -> T {
    do {
        return try body()
    } catch {
        throw TrainingError.metal(describeMetalError(error))
    }
}

/// Human-readable text for any error raised by a Metal call.
func describeMetalError(_ error: any Error) -> String {
    if let metal = error as? MetalError {
        switch metal {
        case .noDevice:
            return "no Metal device"
        case .shaderCompilation(let message):
            return "shader compilation failed: \(message)"
        case .bufferAllocation:
            return "buffer allocation failed"
        case .commandFailed(let message):
            return message
        }
    }
    return String(describing: error)
}
