import Foundation
import Testing
import SplatCore
@testable import SplatIO

// Autosave files, recovery selection and the optimizer state inside a project (v1.5). Pure, runs on Linux.

struct AutosaveStoreTests {

    private let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)

    private func makeCloud(count: Int, shDegree: Int = 1) throws -> GaussianCloud {
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        return try GaussianCloud(
            shDegree: shDegree,
            positions: (0..<count * 3).map { Float($0) * 0.1 },
            logScales: [Float](repeating: -2, count: count * 3),
            rotations: (0..<count * 4).map { Float($0 % 4 == 0 ? 1 : 0) },
            opacityLogits: [Float](repeating: 0.5, count: count),
            sh: (0..<count * k * 3).map { Float($0 % 5) * 0.01 })
    }

    private func makeMoments(count: Int, shDegree: Int) -> GaussianMoments {
        let layout = OptimizerState.groupFloats(shDegree: shDegree)
        return GaussianMoments(positions: [Float](repeating: 0.1, count: count * layout[0]),
                               logScales: [Float](repeating: 0.2, count: count * layout[1]),
                               rotations: [Float](repeating: 0.3, count: count * layout[2]),
                               opacityLogits: [Float](repeating: 0.4, count: count * layout[3]),
                               sh: [Float](repeating: 0.5, count: count * layout[4]))
    }

    /// A consistent snapshot of `count` Gaussians at `iteration` for the default configuration.
    private func makeSnapshot(count: Int, iteration: Int, date: Date) throws -> AutosaveSnapshot {
        let hash = OptimizerState.configurationHash(TrainingConfiguration())
        let optimizer = OptimizerState(gaussianCount: count, shDegree: 1, iteration: iteration, adamStep: iteration,
                                       configurationHash: hash,
                                       firstMoment: makeMoments(count: count, shDegree: 1),
                                       secondMoment: makeMoments(count: count, shDegree: 1))
        let manifest = AutosaveManifest(iteration: iteration, date: date, gaussianCount: count,
                                        configurationHash: AutosaveManifest.hashText(hash))
        return AutosaveSnapshot(manifest: manifest, cloud: try makeCloud(count: count), optimizer: optimizer)
    }

    private func temporaryPackage() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "autosave-test-\(UUID().uuidString).gsproj", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test
    func writeThenReadRoundTrips() throws {
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        let snapshot = try makeSnapshot(count: 4, iteration: 2_000, date: fixedDate)
        try AutosaveStore.write(snapshot, packageURL: package)

        let read = try #require(try AutosaveStore.read(packageURL: package))
        #expect(read.manifest == snapshot.manifest)
        #expect(read.cloud.count == 4)
        #expect(read.cloud.positions == snapshot.cloud.positions)
        #expect(read.optimizer == snapshot.optimizer)
    }

    @Test
    func readWithoutAutosaveReturnsNil() throws {
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        #expect(try AutosaveStore.read(packageURL: package) == nil)
        #expect(try AutosaveStore.manifest(packageURL: package) == nil)
    }

    @Test
    func inconsistentSnapshotIsNotWritten() throws {
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        var snapshot = try makeSnapshot(count: 4, iteration: 10, date: fixedDate)
        snapshot.manifest.gaussianCount = 5
        #expect(throws: AutosaveError.self) {
            try AutosaveStore.write(snapshot, packageURL: package)
        }
        #expect(try AutosaveStore.manifest(packageURL: package) == nil)
    }

    @Test
    func truncatedOptimizerFileIsCorrupt() throws {
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        try AutosaveStore.write(try makeSnapshot(count: 3, iteration: 10, date: fixedDate), packageURL: package)
        let generation = try #require(try AutosaveStore.manifest(packageURL: package)).generation
        let optimizerURL = package.appending(path: AutosaveLayout.optimizer(generation: generation))
        let data = try Data(contentsOf: optimizerURL)
        try data.prefix(data.count / 2).write(to: optimizerURL)
        do {
            _ = try AutosaveStore.read(packageURL: package)
            Issue.record("a truncated optimizer file was read")
        } catch let error as AutosaveError {
            guard case .corrupt = error else {
                Issue.record("expected corruption, got \(error)")
                return
            }
        }
    }

    @Test
    func crashBeforeTheManifestKeepsThePreviousAutosave() throws {
        // The new generation's data files are written, then the process dies before the manifest is replaced.
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        let previous = try makeSnapshot(count: 3, iteration: 10, date: fixedDate)
        try AutosaveStore.write(previous, packageURL: package)
        let next = try makeSnapshot(count: 6, iteration: 20, date: fixedDate)
        let checkpoints = package.appending(path: "checkpoints", directoryHint: .isDirectory)
        try PLYWriter.write(next.cloud).write(to: package.appending(path: AutosaveLayout.cloud(generation: next.manifest.generation)))
        try next.optimizer.encode().write(to: package.appending(path: AutosaveLayout.optimizer(generation: next.manifest.generation)))
        // The previous autosave is still complete and is what the reader returns.
        let read = try #require(try AutosaveStore.read(packageURL: package))
        #expect(read.manifest == previous.manifest)
        #expect(read.cloud.count == 3)
        #expect(FileManager.default.fileExists(atPath: checkpoints.path))
    }

    @Test
    func manifestOfARetiredGenerationIsCorrupt() throws {
        // Once a newer generation is committed, the older one is removed; a manifest that names it cannot be read.
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        let old = try makeSnapshot(count: 3, iteration: 10, date: fixedDate)
        try AutosaveStore.write(old, packageURL: package)
        try AutosaveStore.write(try makeSnapshot(count: 6, iteration: 20, date: fixedDate), packageURL: package)
        try AutosaveStore.encodeManifest(old.manifest).write(to: package.appending(path: AutosaveLayout.manifest))
        #expect(throws: AutosaveError.self) {
            _ = try AutosaveStore.read(packageURL: package)
        }
    }

    @Test
    func removeDeletesEveryAutosaveFile() throws {
        let package = try temporaryPackage()
        defer { try? FileManager.default.removeItem(at: package) }
        let checkpoints = package.appending(path: "checkpoints", directoryHint: .isDirectory)
        try AutosaveStore.write(try makeSnapshot(count: 2, iteration: 10, date: fixedDate), packageURL: package)
        try AutosaveStore.write(try makeSnapshot(count: 2, iteration: 20, date: fixedDate), packageURL: package)
        // Only the newest generation's three files exist before the removal.
        let names = try FileManager.default.contentsOfDirectory(atPath: checkpoints.path).filter { $0.hasPrefix("autosave") }
        #expect(names.count == 3)
        try AutosaveStore.remove(packageURL: package)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: checkpoints.path).filter { $0.hasPrefix("autosave") }
        #expect(remaining.isEmpty)
        // Removing again is not an error.
        try AutosaveStore.remove(packageURL: package)
    }

    @Test
    func unknownManifestFormatIsCorrupt() throws {
        var manifest = try makeSnapshot(count: 1, iteration: 1, date: fixedDate).manifest
        manifest.formatVersion = 99
        let data = try AutosaveStore.encodeManifest(manifest)
        #expect(throws: AutosaveError.self) {
            _ = try AutosaveStore.decodeManifest(data)
        }
    }

    // MARK: - Recovery selection

    @Test
    func recoveryPrefersAutosaveOnlyWhenItIsNewer() {
        let checkpointDate = Date(timeIntervalSince1970: 1_700_000_000)
        let checkpoint = CheckpointRecord(iteration: 1_000, gaussianCount: 9, psnr: nil, date: checkpointDate)
        func autosave(iteration: Int, date: Date) -> AutosaveManifest {
            AutosaveManifest(iteration: iteration, date: date, gaussianCount: 9, configurationHash: "0")
        }

        // No autosave: the checkpoint opens.
        #expect(RecoverySelection.choose(autosave: nil, latestCheckpoint: checkpoint) == .checkpoint)
        // An autosave without progress is not offered.
        #expect(RecoverySelection.choose(autosave: autosave(iteration: 0, date: checkpointDate.addingTimeInterval(60)),
                                         latestCheckpoint: checkpoint) == .checkpoint)
        // A newer autosave is offered, even at a lower iteration (a new run that crashed early).
        #expect(RecoverySelection.choose(autosave: autosave(iteration: 300, date: checkpointDate.addingTimeInterval(60)),
                                         latestCheckpoint: checkpoint) == .autosave)
        // An older autosave is stale: the checkpoint was stored after it.
        #expect(RecoverySelection.choose(autosave: autosave(iteration: 2_000, date: checkpointDate.addingTimeInterval(-60)),
                                         latestCheckpoint: checkpoint) == .checkpoint)
        // Same date: the autosave is offered only when it covers more iterations.
        #expect(RecoverySelection.choose(autosave: autosave(iteration: 2_000, date: checkpointDate),
                                         latestCheckpoint: checkpoint) == .autosave)
        #expect(RecoverySelection.choose(autosave: autosave(iteration: 1_000, date: checkpointDate),
                                         latestCheckpoint: checkpoint) == .checkpoint)
        // Without a checkpoint, any autosave with progress is offered.
        #expect(RecoverySelection.choose(autosave: autosave(iteration: 50, date: checkpointDate),
                                         latestCheckpoint: nil) == .autosave)
    }

    // MARK: - Project bundle

    private func sampleState(count: Int) -> OptimizerState {
        OptimizerState(gaussianCount: count, shDegree: 1, iteration: 500, adamStep: 500,
                       configurationHash: 42,
                       firstMoment: makeMoments(count: count, shDegree: 1),
                       secondMoment: makeMoments(count: count, shDegree: 1))
    }

    @Test
    func checkpointStoresOptimizerStateAndRecordsIt() throws {
        let cloud = try makeCloud(count: 3)
        let record = CheckpointRecord(iteration: 500, gaussianCount: 3, psnr: nil, date: fixedDate)
        let withState = ProjectBundle.withCheckpoint(ProjectContents(manifest: ProjectManifest(), files: [:]),
                                                     cloud: cloud, record: record, optimizer: sampleState(count: 3))
        #expect(withState.manifest.checkpoints.last?.hasOptimizerState == true)
        #expect(try ProjectBundle.optimizerState(withState) == sampleState(count: 3))

        // A checkpoint without state replaces the previous state and says so.
        let withoutState = ProjectBundle.withCheckpoint(withState, cloud: cloud, record: record)
        #expect(withoutState.manifest.checkpoints.last?.hasOptimizerState == nil)
        #expect(withoutState.files[ProjectLayout.optimizerState] == nil)
        #expect(try ProjectBundle.optimizerState(withoutState) == nil)
    }

    @Test
    func optimizerStateRoundTripsThroughPackageEncoding() throws {
        let cloud = try makeCloud(count: 3)
        let record = CheckpointRecord(iteration: 500, gaussianCount: 3, psnr: nil, date: fixedDate)
        let contents = ProjectBundle.withCheckpoint(ProjectContents(manifest: ProjectManifest(), files: [:]),
                                                    cloud: cloud, record: record, optimizer: sampleState(count: 3))
        let encoded = try ProjectBundle.encode(contents)
        #expect(encoded[ProjectLayout.optimizerState] != nil)
        let decoded = try ProjectBundle.decode(files: encoded)
        #expect(try ProjectBundle.optimizerState(decoded) == sampleState(count: 3))
        #expect(decoded.manifest.checkpoints.last?.hasOptimizerState == true)
    }

    @Test
    func discardOptimizerStateClearsFileAndFlag() throws {
        let cloud = try makeCloud(count: 3)
        let record = CheckpointRecord(iteration: 500, gaussianCount: 3, psnr: nil, date: fixedDate)
        let contents = ProjectBundle.withCheckpoint(ProjectContents(manifest: ProjectManifest(), files: [:]),
                                                    cloud: cloud, record: record, optimizer: sampleState(count: 3))
        let discarded = ProjectBundle.withoutOptimizerState(contents)
        #expect(discarded.files[ProjectLayout.optimizerState] == nil)
        #expect(discarded.manifest.checkpoints.last?.hasOptimizerState == false)
        #expect(try ProjectBundle.optimizerState(discarded) == nil)
    }

    @Test
    func corruptOptimizerFileIsReportedAsCorruptCheckpoint() throws {
        var contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        contents.files[ProjectLayout.optimizerState] = Data("not a state".utf8)
        #expect(throws: ProjectError.self) {
            _ = try ProjectBundle.optimizerState(contents)
        }
    }
}
