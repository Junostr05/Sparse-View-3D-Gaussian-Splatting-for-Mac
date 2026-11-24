#if os(macOS)
import Foundation
import Network
import SplatCore
import SplatIO
import SplatWorkProtocol

/// Computes depth maps on a companion device (an iPhone on the same network) and returns them per view.
///
/// The companion is found with Bonjour (`_sparsesplat-work._tcp`). When none answers within the timeout, the result is
/// empty, and the caller uses its local estimator. A view whose request fails is left out, so one bad response never
/// stops the run.
public enum RemoteDepthWorker {
    public static let serviceType: String = "_sparsesplat-work._tcp"

    /// Depth maps for the views, keyed by view name. Empty when no companion is found.
    /// - Parameters:
    ///   - views: The training views to compute.
    ///   - model: The companion's model name, for example `depth-anything-3-base`.
    ///   - discoveryTimeout: How long to look for a companion, in seconds.
    public static func depthMaps(for views: [TrainingView], model: String,
                                 discoveryTimeout: Double = 5) async -> [String: DepthMap] {
        guard !views.isEmpty, let endpoint = await findCompanion(timeout: discoveryTimeout) else {
            return [:]
        }
        guard let connection = await open(endpoint) else {
            return [:]
        }
        defer {
            connection.cancel()
        }
        var maps: [String: DepthMap] = [:]
        for view in views {
            let request = WorkProtocol.DepthRequest(id: view.name, name: view.name, model: model,
                                                    width: view.image.width, height: view.image.height,
                                                    rgb: rgb8(view.image))
            guard let reply = try? await exchange(WorkProtocol.encode(.depthRequest(request)), on: connection),
                  case .depthResponse(let id, let map) = reply, id == view.name else {
                continue
            }
            maps[view.name] = map
        }
        return maps
    }

    // MARK: - Pixels

    /// The image as RGB8 bytes, row-major. Values are clamped to 0...1 before the conversion.
    static func rgb8(_ image: RGBImage) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(image.pixels.count * 3)
        for pixel in image.pixels {
            bytes.append(byte(pixel.x))
            bytes.append(byte(pixel.y))
            bytes.append(byte(pixel.z))
        }
        return bytes
    }

    private static func byte(_ value: Float) -> UInt8 {
        let clamped: Float = min(max(value.isNaN ? 0 : value, 0), 1)
        return UInt8((clamped * 255).rounded())
    }

    // MARK: - Network

    /// The first companion Bonjour finds, or `nil` after `timeout` seconds.
    private static func findCompanion(timeout: Double) async -> NWEndpoint? {
        await withCheckedContinuation { (continuation: CheckedContinuation<NWEndpoint?, Never>) in
            let once = OnceBox()
            let browser = NWBrowser(for: .bonjour(type: serviceType, domain: nil), using: .tcp)
            browser.browseResultsChangedHandler = { results, _ in
                guard let result = results.first else {
                    return
                }
                if once.claim() {
                    browser.cancel()
                    continuation.resume(returning: result.endpoint)
                }
            }
            browser.start(queue: .global(qos: .userInitiated))
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + timeout) {
                if once.claim() {
                    browser.cancel()
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    /// A connection to the endpoint once it is ready, or `nil` when it fails.
    private static func open(_ endpoint: NWEndpoint) async -> NWConnection? {
        await withCheckedContinuation { (continuation: CheckedContinuation<NWConnection?, Never>) in
            let connection = NWConnection(to: endpoint, using: .tcp)
            let once = OnceBox()
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if once.claim() {
                        continuation.resume(returning: connection)
                    }
                case .failed, .cancelled:
                    if once.claim() {
                        continuation.resume(returning: nil)
                    }
                default:
                    break
                }
            }
            connection.start(queue: .global(qos: .userInitiated))
        }
    }

    /// Sends one frame and reads the reply frame.
    private static func exchange(_ frame: Data, on connection: NWConnection) async throws -> WorkProtocol.Message {
        try await send(frame, on: connection)
        let header: Data = try await receive(exactly: WorkProtocol.headerSize, on: connection)
        let length: Int = try WorkProtocol.payloadLength(header: header)
        let payload: Data = try await receive(exactly: length, on: connection)
        return try WorkProtocol.decode(header + payload)
    }

    private static func send(_ data: Data, on connection: NWConnection) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            })
        }
    }

    /// Reads exactly `count` bytes, across as many network reads as needed.
    private static func receive(exactly count: Int, on connection: NWConnection) async throws -> Data {
        var data = Data()
        while data.count < count {
            let chunk: Data = try await receiveChunk(maximum: count - data.count, on: connection)
            data.append(chunk)
        }
        return data
    }

    private static func receiveChunk(maximum: Int, on connection: NWConnection) async throws -> Data {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
            connection.receive(minimumIncompleteLength: 1, maximumLength: maximum) { data, _, isComplete, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let data, !data.isEmpty {
                    continuation.resume(returning: data)
                } else if isComplete {
                    continuation.resume(throwing: WorkProtocol.Failure.truncated)
                } else {
                    continuation.resume(returning: Data())
                }
            }
        }
    }
}

/// Resumes a continuation once: the first caller of `claim` wins.
private final class OnceBox: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if claimed {
            return false
        }
        claimed = true
        return true
    }
}
#endif
