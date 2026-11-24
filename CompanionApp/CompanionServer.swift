import Foundation
import Network
import SplatCore
import SplatWorkProtocol

/// Listens for work from the Mac: advertises `_sparsesplat-work._tcp` on the local network and answers depth requests.
/// Requests are handled one at a time per connection; each answer is a framed `WorkProtocol` message.
nonisolated final class CompanionServer: @unchecked Sendable {
    static let serviceType = "_sparsesplat-work._tcp"

    private let depth: CompanionDepth
    private let modelName: String
    private let listener: NWListener
    /// Called on the main actor with a status line whenever something changes.
    private let onStatus: @Sendable (String) -> Void

    init(depth: CompanionDepth, modelName: String, onStatus: @escaping @Sendable (String) -> Void) throws {
        self.depth = depth
        self.modelName = modelName
        self.onStatus = onStatus
        listener = try NWListener(using: .tcp)
        listener.service = NWListener.Service(type: CompanionServer.serviceType)
        listener.newConnectionHandler = { [weak self] connection in
            self?.serve(connection)
        }
        listener.stateUpdateHandler = { state in
            if case .ready = state {
                onStatus("Waiting for the Mac on the local network.")
            }
        }
    }

    func start() {
        listener.start(queue: .global(qos: .userInitiated))
    }

    func stop() {
        listener.cancel()
    }

    // MARK: - Connections

    private func serve(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInitiated))
        onStatus("The Mac is connected.")
        Task {
            while true {
                do {
                    let header = try await receive(exactly: WorkProtocol.headerSize, on: connection)
                    let length = try WorkProtocol.payloadLength(header: header)
                    let payload = try await receive(exactly: length, on: connection)
                    let reply = await answer(to: try WorkProtocol.decode(header + payload))
                    try await send(WorkProtocol.encode(reply), on: connection)
                } catch {
                    onStatus("Connection ended.")
                    connection.cancel()
                    return
                }
            }
        }
    }

    private func answer(to message: WorkProtocol.Message) async -> WorkProtocol.Message {
        guard case .depthRequest(let request) = message else {
            return .failure(id: "", message: "this companion only answers depth requests")
        }
        guard request.model == modelName else {
            return .failure(id: request.id, message: "this companion runs \(modelName), not \(request.model)")
        }
        do {
            let map = try depth.depth(rgb: request.rgb, width: request.width, height: request.height)
            onStatus("Answered \(request.name).")
            return .depthResponse(id: request.id, map: map)
        } catch {
            return .failure(id: request.id, message: String(describing: error))
        }
    }

    // MARK: - Framed I/O

    private func send(_ data: Data, on connection: NWConnection) async throws {
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

    private func receive(exactly count: Int, on connection: NWConnection) async throws -> Data {
        var data = Data()
        while data.count < count {
            let chunk: Data = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
                connection.receive(minimumIncompleteLength: 1, maximumLength: count - data.count) { chunk, _, isComplete, error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else if let chunk, !chunk.isEmpty {
                        continuation.resume(returning: chunk)
                    } else if isComplete {
                        continuation.resume(throwing: WorkProtocol.Failure.truncated)
                    } else {
                        continuation.resume(returning: Data())
                    }
                }
            }
            data.append(chunk)
        }
        return data
    }
}
