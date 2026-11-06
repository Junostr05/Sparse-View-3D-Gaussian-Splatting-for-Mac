import Foundation
import Testing
import SplatCore
@testable import SplatIO

private func littleEndianUInt32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
    var value: UInt32 = 0
    for index in 0..<4 {
        value |= UInt32(bytes[offset + index]) << UInt32(8 * index)
    }
    return value
}

private func sampleMap(width: Int, height: Int, kind: DepthKind,
                       confidence: Bool, estimated: Bool) -> DepthMap {
    var values: [Float] = []
    for index in 0..<(width * height) {
        values.append(Float(index) * 0.25 + 0.5)
    }
    var confidenceValues: [UInt8]? = nil
    if confidence {
        var bytes: [UInt8] = []
        for index in 0..<(width * height) {
            bytes.append(UInt8(index % 3))
        }
        confidenceValues = bytes
    }
    return DepthMap(width: width, height: height, kind: kind, values: values,
                    confidence: confidenceValues, isEstimated: estimated)
}

@Suite struct DepthMapCodecTests {

    @Test func roundTripsAllKindsWithAndWithoutConfidence() throws {
        let kinds: [DepthKind] = [.metric, .relativeDepth, .relativeInverseDepth]
        for kind in kinds {
            for confidence in [false, true] {
                let map = sampleMap(width: 5, height: 3, kind: kind,
                                    confidence: confidence, estimated: false)
                let decoded = try DepthMapCodec.decode(DepthMapCodec.encode(map))
                #expect(decoded == map)
            }
        }
    }

    @Test func headerIsTwentyFourBytesLittleEndian() {
        let map = DepthMap(width: 2, height: 1, kind: .metric, values: [1, 2])
        let data = DepthMapCodec.encode(map)
        let bytes = [UInt8](data)

        #expect(data.count == 24 + 8)
        #expect(Array(bytes[0..<4]) == [0x53, 0x44, 0x50, 0x54])   // "SDPT"
        #expect(bytes[4] == 1 && bytes[5] == 0)                     // version 1, UInt16 LE
        #expect(bytes[6] == 0)                                      // kind metric
        #expect(bytes[7] == 0)                                      // flags: no confidence, not estimated
        #expect(littleEndianUInt32(bytes, at: 8) == 2)              // width
        #expect(littleEndianUInt32(bytes, at: 12) == 1)             // height
        for index in 16..<24 {
            #expect(bytes[index] == 0)                              // reserved
        }
        #expect(Float(bitPattern: littleEndianUInt32(bytes, at: 24)) == 1)
        #expect(Float(bitPattern: littleEndianUInt32(bytes, at: 28)) == 2)
    }

    @Test func rejectsBadMagicVersionKindAndTruncation() throws {
        let good = [UInt8](DepthMapCodec.encode(
            DepthMap(width: 2, height: 1, kind: .metric, values: [1, 2])))

        var badMagic = good
        badMagic[0] = 0x58
        #expect(throws: DepthMapError.badMagic) {
            try DepthMapCodec.decode(Data(badMagic))
        }

        var badVersion = good
        badVersion[4] = 7
        #expect(throws: DepthMapError.unsupportedVersion(7)) {
            try DepthMapCodec.decode(Data(badVersion))
        }

        var badKind = good
        badKind[6] = 9
        #expect(throws: DepthMapError.unknownKind(9)) {
            try DepthMapCodec.decode(Data(badKind))
        }

        let truncated = Array(good[0..<(good.count - 1)])
        #expect(throws: DepthMapError.truncated(expected: 32, actual: 31)) {
            try DepthMapCodec.decode(Data(truncated))
        }
    }

    @Test func estimatedFlagRoundTrips() throws {
        let map = sampleMap(width: 3, height: 2, kind: .relativeInverseDepth,
                            confidence: true, estimated: true)
        let bytes = [UInt8](DepthMapCodec.encode(map))
        #expect(bytes[7] == 0b11)
        let decoded = try DepthMapCodec.decode(Data(bytes))
        #expect(decoded.isEstimated)
        #expect(decoded == map)
    }
}

@Suite struct DepthProjectStorageTests {

    @Test func arkitImportStoresDepthFiles() {
        let depth = DepthMapCodec.encode(
            DepthMap(width: 1, height: 1, kind: .metric, values: [2]))
        let imported = ProjectBundle.importARKit(
            capture: Data("{}".utf8),
            imageFiles: ["a.ppm": Data([1])],
            depthFiles: ["a.ppm": depth],
            into: ProjectContents(manifest: ProjectManifest(), files: [:]))

        #expect(imported.files[ProjectLayout.depthPath(forImage: "a.ppm")] == depth)
        #expect(ProjectLayout.depthPath(forImage: "a.ppm") == "depth/a.ppm.sdpt")
    }

    @Test func arkitImportWithoutDepthFilesClearsOldDepth() {
        var base = ProjectContents(manifest: ProjectManifest(), files: [:])
        base.files[ProjectLayout.depthPath(forImage: "old.ppm")] = Data([1, 2, 3])
        let imported = ProjectBundle.importARKit(capture: Data("{}".utf8),
                                                 imageFiles: [:],
                                                 into: base)
        #expect(imported.files[ProjectLayout.depthPath(forImage: "old.ppm")] == nil)
    }

    @Test func withDepthMapsRoundTripsThroughProject() throws {
        let map = sampleMap(width: 4, height: 2, kind: .metric, confidence: true, estimated: false)
        let contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        let stored = ProjectBundle.withDepthMaps(contents, ["view.ppm": map])
        let loaded = ProjectBundle.depthMaps(stored)
        #expect(loaded.failures.isEmpty)
        #expect(loaded.maps == ["view.ppm": map])
    }

    @Test func depthMapsReportsCorruptFilesByName() {
        var contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        contents.files[ProjectLayout.depthPath(forImage: "good.ppm")] =
            DepthMapCodec.encode(DepthMap(width: 1, height: 1, kind: .metric, values: [3]))
        contents.files[ProjectLayout.depthPath(forImage: "bad.ppm")] = Data("nope".utf8)
        contents.files["images/good.ppm"] = Data([0])

        let loaded = ProjectBundle.depthMaps(contents)
        #expect(loaded.maps.keys.sorted() == ["good.ppm"])
        #expect(loaded.failures.keys.sorted() == ["bad.ppm"])
        #expect(loaded.failures["bad.ppm"] == .truncated(expected: 24, actual: 4))
    }

    @Test func checkpointRecordDecodesWithoutHeldOutFields() throws {
        let json = Data(#"{"iteration":10,"gaussianCount":3,"date":"2026-11-06T00:00:00Z"}"#.utf8)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let record = try decoder.decode(CheckpointRecord.self, from: json)
        #expect(record.iteration == 10)
        #expect(record.gaussianCount == 3)
        #expect(record.psnr == nil)
        #expect(record.heldOutView == nil)
        #expect(record.heldOutPSNR == nil)
        #expect(record.heldOutSSIM == nil)
        #expect(record.bestIteration == nil)
    }
}
