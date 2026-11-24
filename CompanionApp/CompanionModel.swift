import Foundation
import Observation
import UIKit

/// The companion's state for the screen: loads the depth model from the app bundle and runs the server.
@MainActor
@Observable
final class CompanionModel {
    /// The model package name inside the app bundle (add it to the companion target's resources).
    static let bundledModel: String = "DepthAnything3BaseQ8"
    static let modelName: String = "depth-anything-3-base"

    private(set) var status: String = "Not started."
    private var server: CompanionServer?

    func start() {
        guard server == nil else { return }
        guard let url = Bundle.main.url(forResource: Self.bundledModel, withExtension: "mlpackage") else {
            status = "The depth model is not in the app. Add \(Self.bundledModel).mlpackage to the app."
            return
        }
        // The screen stays on while the companion works: iOS suspends a backgrounded app.
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Loading the depth model…"
        Task {
            do {
                let depth = try await CompanionDepth(packageURL: url)
                let started = try CompanionServer(depth: depth, modelName: Self.modelName) { [weak self] text in
                    Task { @MainActor in self?.status = text }
                }
                started.start()
                self.server = started
                self.status = "Waiting for the Mac on the local network."
            } catch {
                self.status = "Could not start: \(String(describing: error))"
            }
        }
    }

    func stop() {
        server?.stop()
        server = nil
        UIApplication.shared.isIdleTimerDisabled = false
        status = "Stopped."
    }
}
