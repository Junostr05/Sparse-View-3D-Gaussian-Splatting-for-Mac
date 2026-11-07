import Foundation
import Testing
import SplatCore
import SplatIO

private func editCloud(count: Int) throws -> GaussianCloud {
    try GaussianCloud(shDegree: 0,
                      positions: (0..<count * 3).map { Float($0) },
                      logScales: [Float](repeating: 0, count: count * 3),
                      rotations: (0..<count).flatMap { _ in [Float(1), 0, 0, 0] },
                      opacityLogits: [Float](repeating: 0, count: count),
                      sh: [Float](repeating: 0, count: count * 3))
}

/// A project with one checkpoint of `gaussianCount` Gaussians and the given edits.
private func projectWithCheckpoint(gaussianCount: Int) throws -> ProjectContents {
    let cloud = try editCloud(count: gaussianCount)
    let base = ProjectContents(manifest: ProjectManifest(), files: [:])
    return ProjectBundle.withCheckpoint(base, cloud: cloud,
                                        record: CheckpointRecord(iteration: 10,
                                                                 gaussianCount: gaussianCount,
                                                                 psnr: nil,
                                                                 date: Date(timeIntervalSince1970: 0)))
}

@Suite struct DeletionListTests {
    @Test func roundTripsSortedIndices() throws {
        let indices: [UInt32] = [0, 1, 2, 127, 128, 16_383, 16_384, 1_000_000, UInt32.max - 1, UInt32.max]
        let data = DeletionListCodec.encode(indices)
        let decoded = try DeletionListCodec.decode(data, maxIndex: Int(UInt32.max) + 1)
        #expect(decoded == indices)
    }

    @Test func emptyListRoundTrips() throws {
        #expect(DeletionListCodec.encode([]).isEmpty)
        #expect(try DeletionListCodec.decode(Data(), maxIndex: 0).isEmpty)
    }

    @Test func deltaEncodingIsCompact() {
        // 0, 1, 2, … each delta is 1 and fits in one byte.
        let data = DeletionListCodec.encode([0, 1, 2, 3])
        #expect([UInt8](data) == [0, 1, 1, 1])
    }

    @Test func rejectsIndexBeyondMaxIndex() {
        let data = DeletionListCodec.encode([5, 9])
        #expect(throws: DeletionListError.indexOutOfRange(9)) {
            try DeletionListCodec.decode(data, maxIndex: 9)
        }
        #expect(throws: DeletionListError.indexOutOfRange(5)) {
            try DeletionListCodec.decode(DeletionListCodec.encode([5]), maxIndex: 5)
        }
    }

    @Test func rejectsTruncatedVarint() {
        // 0x80 announces a continuation byte that never comes.
        #expect(throws: DeletionListError.truncated) {
            try DeletionListCodec.decode(Data([0x80]), maxIndex: 100)
        }
    }

    @Test func rejectsZeroDeltaAsNotAscending() {
        // Index 3, then delta 0 gives index 3 again.
        #expect(throws: DeletionListError.notAscending) {
            try DeletionListCodec.decode(Data([3, 0]), maxIndex: 100)
        }
    }

    @Test func rejectsVarintOverflow() {
        // Five bytes with the top group set past 32 bits.
        let tooBig = Data([0xff, 0xff, 0xff, 0xff, 0x7f])
        #expect(throws: DeletionListError.overflow) {
            try DeletionListCodec.decode(tooBig, maxIndex: Int.max)
        }
        // Six continuation bytes.
        let endless = Data([0x80, 0x80, 0x80, 0x80, 0x80, 0x01])
        #expect(throws: DeletionListError.overflow) {
            try DeletionListCodec.decode(endless, maxIndex: Int.max)
        }
    }

    @Test func rejectsSumOverflow() {
        // UInt32.max, then one more: the running sum exceeds 32 bits.
        let data = DeletionListCodec.encode([UInt32.max]) + Data([1])
        #expect(throws: DeletionListError.overflow) {
            try DeletionListCodec.decode(data, maxIndex: Int.max)
        }
    }

    @Test func negativeMaxIndexRejectsEverything() {
        #expect(throws: DeletionListError.indexOutOfRange(0)) {
            try DeletionListCodec.decode(Data([0]), maxIndex: -1)
        }
    }
}

@Suite struct ProjectEditsTests {
    @Test func deletionsRoundTripThroughPackage() throws {
        let contents = try projectWithCheckpoint(gaussianCount: 20)
        let edited = ProjectBundle.withDeletions(contents, baseGaussianCount: 20, deleted: [3, 7, 19])

        #expect(edited.manifest.edits == EditRecord(baseGaussianCount: 20, deletedCount: 3))
        #expect(edited.files[ProjectLayout.deletedIndices] != nil)

        // Through the package encoding and back.
        let files = try ProjectBundle.encode(edited)
        #expect(files[ProjectLayout.deletedIndices] != nil)
        let decoded = try ProjectBundle.decode(files: files)
        #expect(decoded == edited)

        let stored = try ProjectBundle.deletions(decoded)
        #expect(stored?.baseGaussianCount == 20)
        #expect(stored?.deleted == [3, 7, 19])
    }

    @Test func emptyListRemovesFileAndRecord() throws {
        let contents = try projectWithCheckpoint(gaussianCount: 5)
        let edited = ProjectBundle.withDeletions(contents, baseGaussianCount: 5, deleted: [1])
        let cleared = ProjectBundle.withDeletions(edited, baseGaussianCount: 5, deleted: [])

        #expect(cleared.manifest.edits == nil)
        #expect(cleared.files[ProjectLayout.deletedIndices] == nil)
        #expect(try ProjectBundle.deletions(cleared) == nil)
    }

    @Test func duplicatesAreRemovedAndIndicesSorted() throws {
        let contents = try projectWithCheckpoint(gaussianCount: 10)
        let edited = ProjectBundle.withDeletions(contents, baseGaussianCount: 10, deleted: [8, 2, 8, 2])
        #expect(edited.manifest.edits == EditRecord(baseGaussianCount: 10, deletedCount: 2))
        #expect(try ProjectBundle.deletions(edited)?.deleted == [2, 8])
    }

    @Test func noDeletionsWhenNothingStored() throws {
        let contents = try projectWithCheckpoint(gaussianCount: 4)
        #expect(try ProjectBundle.deletions(contents) == nil)
    }

    @Test func newCheckpointDropsDeletions() throws {
        let contents = try projectWithCheckpoint(gaussianCount: 6)
        let edited = ProjectBundle.withDeletions(contents, baseGaussianCount: 6, deleted: [0, 5])
        let retrained = ProjectBundle.withCheckpoint(edited,
                                                     cloud: try editCloud(count: 6),
                                                     record: CheckpointRecord(iteration: 20,
                                                                              gaussianCount: 6,
                                                                              psnr: nil,
                                                                              date: Date(timeIntervalSince1970: 1)))
        #expect(retrained.manifest.edits == nil)
        #expect(retrained.files[ProjectLayout.deletedIndices] == nil)
        #expect(try ProjectBundle.deletions(retrained) == nil)
    }

    @Test func importDropsDeletions() throws {
        let contents = try projectWithCheckpoint(gaussianCount: 6)
        let edited = ProjectBundle.withDeletions(contents, baseGaussianCount: 6, deleted: [4])
        let imported = ProjectBundle.importARKit(capture: Data("{}".utf8),
                                                 imageFiles: [:],
                                                 into: edited)
        #expect(imported.manifest.edits == nil)
        #expect(imported.files[ProjectLayout.deletedIndices] == nil)
    }

    @Test func staleDeletionsAreIgnoredWhenCheckpointCountDiffers() throws {
        var contents = try projectWithCheckpoint(gaussianCount: 6)
        contents = ProjectBundle.withDeletions(contents, baseGaussianCount: 6, deleted: [1])
        // Pretend the latest checkpoint now has a different count.
        contents.manifest.checkpoints[0].gaussianCount = 7
        #expect(try ProjectBundle.deletions(contents) == nil)
    }

    @Test func corruptDeletionFileThrows() throws {
        var contents = try projectWithCheckpoint(gaussianCount: 6)
        contents = ProjectBundle.withDeletions(contents, baseGaussianCount: 6, deleted: [1, 2])
        contents.files[ProjectLayout.deletedIndices] = Data([0x80])
        #expect(throws: ProjectError.self) {
            try ProjectBundle.deletions(contents)
        }
    }

    @Test func indexBeyondBaseCountThrows() throws {
        var contents = try projectWithCheckpoint(gaussianCount: 6)
        contents = ProjectBundle.withDeletions(contents, baseGaussianCount: 6, deleted: [1])
        // Replace with an index that the base count does not contain.
        contents.files[ProjectLayout.deletedIndices] = DeletionListCodec.encode([6])
        #expect(throws: ProjectError.self) {
            try ProjectBundle.deletions(contents)
        }
    }

    @Test func recordWithoutFileThrows() throws {
        var contents = try projectWithCheckpoint(gaussianCount: 6)
        contents = ProjectBundle.withDeletions(contents, baseGaussianCount: 6, deleted: [1])
        contents.files[ProjectLayout.deletedIndices] = nil
        #expect(throws: ProjectError.self) {
            try ProjectBundle.deletions(contents)
        }
    }

    @Test func manifestWithoutEditsDecodesAsNil() throws {
        // A manifest written before `edits` existed has no such key.
        let json = """
        {
          "formatVersion" : 1,
          "imageFiles" : [],
          "training" : \(try encodedDefaultTraining()),
          "checkpoints" : []
        }
        """
        let manifest = try JSONDecoder().decode(ProjectManifest.self, from: Data(json.utf8))
        #expect(manifest.edits == nil)
    }

    @Test func editsRecordRoundTripsThroughManifestJSON() throws {
        var manifest = ProjectManifest()
        manifest.edits = EditRecord(baseGaussianCount: 12, deletedCount: 3)
        let data = try ProjectBundle.encodeManifest(manifest)
        let decoded = try JSONDecoder().decode(ProjectManifest.self, from: data)
        #expect(decoded.edits == EditRecord(baseGaussianCount: 12, deletedCount: 3))
    }
}

/// JSON of the default `TrainingConfiguration`, so the hand-written manifest above stays valid.
private func encodedDefaultTraining() throws -> String {
    let data = try JSONEncoder().encode(TrainingConfiguration())
    return String(decoding: data, as: UTF8.self)
}
