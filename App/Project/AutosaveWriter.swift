import Foundation
import os
import SplatIO

/// Writes the autosaves of one training session into the package, off the main actor. Autosaves are written one at a
/// time, in the order they were submitted. Only the newest pending one is kept: an older autosave that has not started
/// writing is superseded, so a slow disk never queues up the large snapshots.
nonisolated final class AutosaveWriter: Sendable {
    private static let logger = Logger(subsystem: "io.github.junostr05.SparseSplat", category: "autosave")
    private let continuation: AsyncStream<AutosaveSnapshot>.Continuation
    private let task: Task<Void, Never>

    init(packageURL: URL) {
        let (stream, continuation) = AsyncStream.makeStream(of: AutosaveSnapshot.self,
                                                            bufferingPolicy: .bufferingNewest(1))
        self.continuation = continuation
        task = Task.detached(priority: .utility) {
            for await snapshot in stream {
                do {
                    try AutosaveStore.write(snapshot, packageURL: packageURL)
                } catch {
                    // The next autosave writes the same files again. The final checkpoint is unaffected.
                    AutosaveWriter.logger.error("autosave not written: \(String(describing: error), privacy: .public)")
                }
            }
        }
    }

    /// Queues `snapshot` and returns at once: training never waits for the disk.
    nonisolated func submit(_ snapshot: AutosaveSnapshot) {
        continuation.yield(snapshot)
    }

    /// Ends the writer. An autosave already queued is written first.
    func finish() async {
        continuation.finish()
        await task.value
    }
}
