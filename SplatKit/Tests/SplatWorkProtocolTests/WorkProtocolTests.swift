import Foundation
import Testing
import SplatCore
@testable import SplatWorkProtocol

@Suite struct WorkProtocolTests {
    private func sampleMap() -> DepthMap {
        DepthMap(width: 2, height: 2, kind: .metric, values: [1, 2, 3, 4],
                 confidence: nil, isEstimated: true, weights: nil)
    }

    @Test func depthRequestRoundTrips() throws {
        let request = WorkProtocol.DepthRequest(id: "a1", name: "IMG_9014.jpg", model: "depth-anything-3-base",
                                                width: 2, height: 1, rgb: [1, 2, 3, 4, 5, 6])
        let frame = WorkProtocol.encode(.depthRequest(request))
        #expect(try WorkProtocol.decode(frame) == .depthRequest(request))
    }

    @Test func depthResponseRoundTripsItsMap() throws {
        let frame = WorkProtocol.encode(.depthResponse(id: "a1", map: sampleMap()))
        #expect(try WorkProtocol.decode(frame) == .depthResponse(id: "a1", map: sampleMap()))
    }

    @Test func failureRoundTrips() throws {
        let frame = WorkProtocol.encode(.failure(id: "a1", message: "model not loaded"))
        #expect(try WorkProtocol.decode(frame) == .failure(id: "a1", message: "model not loaded"))
    }

    @Test func headerAnnouncesThePayloadLength() throws {
        let frame = WorkProtocol.encode(.failure(id: "a1", message: "x"))
        let length = try WorkProtocol.payloadLength(header: frame.prefix(WorkProtocol.headerSize))
        #expect(length == frame.count - WorkProtocol.headerSize)
    }

    @Test func wrongMagicIsRejected() {
        var frame = WorkProtocol.encode(.failure(id: "a1", message: "x"))
        frame[0] = 0x00
        #expect(throws: WorkProtocol.Failure.badMagic) { try WorkProtocol.decode(frame) }
    }

    @Test func otherVersionIsRejected() {
        var frame = WorkProtocol.encode(.failure(id: "a1", message: "x"))
        frame[4] = 9
        #expect(throws: WorkProtocol.Failure.unsupportedVersion(9)) { try WorkProtocol.decode(frame) }
    }

    @Test func truncatedFrameIsRejected() {
        let frame = WorkProtocol.encode(.failure(id: "a1", message: "a longer message"))
        #expect(throws: WorkProtocol.Failure.truncated) { try WorkProtocol.decode(frame.dropLast(3)) }
    }

    @Test func pixelCountMustMatchTheSize() {
        let request = WorkProtocol.DepthRequest(id: "a1", name: "x.jpg", model: "m", width: 2, height: 2,
                                                rgb: [UInt8](repeating: 0, count: 11))
        let frame = WorkProtocol.encode(.depthRequest(request))
        #expect(throws: WorkProtocol.Failure.pixelCountMismatch(expected: 12, actual: 11)) {
            try WorkProtocol.decode(frame)
        }
    }

    @Test func unknownKindIsRejected() {
        var frame = WorkProtocol.encode(.failure(id: "a1", message: "x"))
        frame[6] = 42
        #expect(throws: WorkProtocol.Failure.unknownKind(42)) { try WorkProtocol.decode(frame) }
    }
}
