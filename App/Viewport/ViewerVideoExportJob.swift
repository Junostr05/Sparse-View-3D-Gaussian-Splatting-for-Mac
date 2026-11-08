import Foundation
import Observation
import SplatCore
import SplatMetal

/// One flight-video export: renders the frames on the main actor, one per step, and writes them to an MP4 through
/// `H264VideoWriter`. The export sheet shows `framesWritten` and the inspector disables the export buttons while
/// `state` is `.running`. Cancelling removes the partial file.
@MainActor
@Observable
final class ViewerVideoExportJob {
    enum State: Equatable {
        case running
        case finished
        case cancelled
        case failed(String)
    }

    let destination: URL
    let size: PixelSize
    let frameCount: Int
    private(set) var state: State = .running
    private(set) var framesWritten: Int = 0
    @ObservationIgnored private var task: Task<Void, Never>?

    /// Starts writing `path` (one view per frame) to `destination` at `size`. An existing file there is replaced.
    init(destination: URL, path: [ViewCamera], size: PixelSize, frameRate: Int,
         renderer: ViewerImageRenderer, model: ViewerModel) {
        self.destination = destination
        self.size = size
        self.frameCount = path.count
        task = Task {
            await self.run(path: path, frameRate: frameRate, renderer: renderer, model: model)
        }
    }

    func cancel() {
        task?.cancel()
    }

    private func run(path: [ViewCamera], frameRate: Int, renderer: ViewerImageRenderer, model: ViewerModel) async {
        var writer: H264VideoWriter?
        do {
            // The first frame must show the final colours of the display mode.
            await model.awaitDisplayColors()
            try Task.checkCancellation()
            try? FileManager.default.removeItem(at: destination)
            let opened = try H264VideoWriter(url: destination, width: size.width, height: size.height,
                                             frameRate: frameRate)
            writer = opened
            for (index, view) in path.enumerated() {
                try Task.checkCancellation()
                try await opened.waitUntilReadyForFrame()
                let texture = try renderer.render(view: view, size: size)
                try opened.append(texture: texture, frameIndex: index)
                framesWritten = index + 1
                await Task.yield()
            }
            try await opened.finish()
            state = .finished
        } catch is CancellationError {
            writer?.cancel()
            try? FileManager.default.removeItem(at: destination)
            state = .cancelled
        } catch {
            writer?.cancel()
            try? FileManager.default.removeItem(at: destination)
            state = .failed(error.localizedDescription)
        }
    }
}
