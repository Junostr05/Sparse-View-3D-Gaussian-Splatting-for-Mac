import Foundation
import Testing
import SplatCore
@testable import SplatIO

// Optimizer-state sidecar (v1.5): binary round trip, corruption cases and validation. Pure, runs on Linux.

struct OptimizerStateTests {

    /// Moments with distinct values for `count` Gaussians of `shDegree`.
    private func moments(count: Int, shDegree: Int, scale: Float) -> GaussianMoments {
        let layout = OptimizerState.groupFloats(shDegree: shDegree)
        func values(_ n: Int) -> [Float] {
            (0..<n).map { Float($0 % 97) * scale - 1.5 }
        }
        return GaussianMoments(positions: values(count * layout[0]),
                               logScales: values(count * layout[1]),
                               rotations: values(count * layout[2]),
                               opacityLogits: values(count * layout[3]),
                               sh: values(count * layout[4]))
    }

    private func makeState(count: Int = 5, shDegree: Int = 1, views: Int = 3,
                           appearance: Bool = true, pose: Bool = true) -> OptimizerState {
        let appearanceBlock: AppearanceOptimizerBlock? = appearance ? AppearanceOptimizerBlock(
            parameters: (0..<views * 6).map { Float($0) * 0.1 },
            firstMoment: (0..<views * 6).map { Float($0) * -0.01 },
            secondMoment: (0..<views * 6).map { Float($0) * 0.001 },
            stepCounts: (0..<views).map { $0 + 7 }) : nil
        let poseBlock: PoseOptimizerBlock? = pose ? PoseOptimizerBlock(
            poses: (0..<views * 12).map { Float($0) * 0.5 - 2 },
            firstMoment: (0..<views * 6).map { Double($0) * 1e-7 },
            secondMoment: (0..<views * 6).map { Double($0) * 3e-9 },
            steps: (0..<views).map { $0 + 3 }) : nil
        return OptimizerState(gaussianCount: count, shDegree: shDegree, iteration: 1_234, adamStep: 2_345,
                              configurationHash: 0xDEAD_BEEF_0123_4567,
                              firstMoment: moments(count: count, shDegree: shDegree, scale: 0.25),
                              secondMoment: moments(count: count, shDegree: shDegree, scale: 0.5),
                              appearance: appearanceBlock, pose: poseBlock)
    }

    @Test
    func roundTripKeepsEveryField() throws {
        let state = makeState()
        let decoded = try OptimizerState(decoding: state.encode())
        #expect(decoded == state)
        #expect(decoded.iteration == 1_234)
        #expect(decoded.adamStep == 2_345)
        #expect(decoded.appearance?.stepCounts == [7, 8, 9])
        #expect(decoded.pose?.steps == [3, 4, 5])
    }

    @Test
    func roundTripWithoutOptionalBlocksAndDegreeThree() throws {
        let state = makeState(count: 3, shDegree: 3, views: 0, appearance: false, pose: false)
        let decoded = try OptimizerState(decoding: state.encode())
        #expect(decoded == state)
        #expect(decoded.appearance == nil)
        #expect(decoded.pose == nil)
    }

    @Test
    func emptyModelRoundTrips() throws {
        let state = makeState(count: 0, shDegree: 0, views: 1, appearance: false, pose: false)
        #expect(try OptimizerState(decoding: state.encode()) == state)
    }

    @Test
    func wrongMagicIsRejected() throws {
        var bytes = [UInt8](makeState().encode())
        bytes[0] = UInt8(ascii: "X")
        #expect(throws: OptimizerStateError.badMagic) {
            _ = try OptimizerState(decoding: Data(bytes))
        }
    }

    @Test
    func unsupportedVersionIsRejected() throws {
        var bytes = [UInt8](makeState().encode())
        bytes[8] = 2   // version field, little-endian UInt32 after the 8-byte magic
        #expect(throws: OptimizerStateError.unsupportedVersion(2)) {
            _ = try OptimizerState(decoding: Data(bytes))
        }
    }

    @Test
    func truncatedFilesAreRejected() throws {
        let full = makeState().encode()
        // Every cut must throw; a cut in the middle of the payload reports truncation with its offset.
        for cut in [0, 1, 7, 8, 12, 40, full.count / 2, full.count - 5, full.count - 4, full.count - 1] {
            #expect(throws: (any Error).self) {
                _ = try OptimizerState(decoding: full.prefix(cut))
            }
        }
        do {
            _ = try OptimizerState(decoding: full.prefix(full.count / 2))
            Issue.record("a truncated file decoded")
        } catch {
            guard case .truncated = error else {
                Issue.record("expected truncation, got \(error)")
                return
            }
        }
    }

    @Test
    func flippedPayloadByteFailsChecksum() throws {
        var bytes = [UInt8](makeState().encode())
        // Inside the first moments, past the header (magic, version, count, degree, layout, counters, flags).
        bytes[bytes.count / 3] ^= 0x40
        #expect(throws: OptimizerStateError.checksumMismatch) {
            _ = try OptimizerState(decoding: Data(bytes))
        }
    }

    @Test
    func inconsistentLayoutIsRejected() throws {
        var bytes = [UInt8](makeState(shDegree: 1).encode())
        // SH degree field sits after magic (8), version (4) and count (8). Degree 2 needs a different SH group size.
        bytes[20] = 2
        do {
            _ = try OptimizerState(decoding: Data(bytes))
            Issue.record("a layout that contradicts the SH degree decoded")
        } catch {
            guard case .inconsistent = error else {
                Issue.record("expected an inconsistent layout, got \(error)")
                return
            }
        }
    }

    @Test
    func trailingBytesAreRejected() throws {
        var bytes = [UInt8](makeState().encode())
        bytes.append(0)
        #expect(throws: (any Error).self) {
            _ = try OptimizerState(decoding: Data(bytes))
        }
    }

    @Test
    func validationNamesEachMismatch() {
        let state = makeState(count: 5, shDegree: 1, views: 3)
        #expect(state.incompatibility(gaussianCount: 5, shDegree: 1, viewCount: 3,
                                      appearance: true, poseRefinement: true) == nil)
        #expect(state.incompatibility(gaussianCount: 4, shDegree: 1, viewCount: 3,
                                      appearance: true, poseRefinement: true)?.contains("5 Gaussians") == true)
        #expect(state.incompatibility(gaussianCount: 5, shDegree: 0, viewCount: 3,
                                      appearance: true, poseRefinement: true)?.contains("SH degree") == true)
        #expect(state.incompatibility(gaussianCount: 5, shDegree: 1, viewCount: 4,
                                      appearance: true, poseRefinement: true)?.contains("views") == true)
        #expect(state.incompatibility(gaussianCount: 5, shDegree: 1, viewCount: 3,
                                      appearance: false, poseRefinement: true) != nil)
        #expect(state.incompatibility(gaussianCount: 5, shDegree: 1, viewCount: 3,
                                      appearance: true, poseRefinement: false) != nil)
        let bare = makeState(appearance: false, pose: false)
        #expect(bare.incompatibility(gaussianCount: 5, shDegree: 1, viewCount: 3,
                                     appearance: true, poseRefinement: false) != nil)
        #expect(bare.incompatibility(gaussianCount: 5, shDegree: 1, viewCount: 3,
                                     appearance: false, poseRefinement: false) == nil)
    }

    @Test
    func configurationHashIsStableAndSensitive() {
        var configuration = TrainingConfiguration()
        let base = OptimizerState.configurationHash(configuration)
        #expect(OptimizerState.configurationHash(configuration) == base)

        // The autosave interval does not change the optimization, so it does not change the hash.
        configuration.autosaveInterval = 500
        #expect(OptimizerState.configurationHash(configuration) == base)

        // Regularizer order is not part of the value: the set is hashed in sorted order.
        var reordered = TrainingConfiguration()
        reordered.regularizers = Set(TrainingConfiguration.defaultRegularizers.reversed())
        #expect(OptimizerState.configurationHash(reordered) == base)

        configuration.learningRates.shDC = 3e-3
        #expect(OptimizerState.configurationHash(configuration) != base)
    }

    @Test
    func autosaveAndCheckpointRecordsDecodeWithoutNewFields() throws {
        // A checkpoint record written before v1.5 has no exact-resume or recovery keys.
        let legacy = """
        {"iteration": 100, "gaussianCount": 7, "date": "2026-11-01T10:00:00Z"}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let record = try decoder.decode(CheckpointRecord.self, from: Data(legacy.utf8))
        #expect(record.hasOptimizerState == nil)
        #expect(record.recovered == nil)
    }

    @Test
    func configurationWithoutAutosaveIntervalDecodesToDefault() throws {
        let encoded = try JSONEncoder().encode(TrainingConfiguration())
        var object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] ?? [:]
        object.removeValue(forKey: "autosaveInterval")
        let stripped = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: stripped)
        #expect(decoded.autosaveInterval == 2_000)
    }
}
