import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU densification and the statistics hook, checked against the CPU reference in SplatCore.
/// Tiny scenes only (CI budget: ≤ 10 s per test). Not verified on Linux: compiled and run on macOS CI only.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPUDensifierTests {

    /// Result of one GPU densify call and the buffers it operated on.
    private struct GPURun {
        let store: GaussianBuffers
        let m: GaussianBuffers
        let v: GaussianBuffers
        let statistics: DensificationStatisticsBuffers
        let summary: DensifySummary
    }

    // MARK: - Helpers

    private func settings() -> RasterSettings {
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        return settings
    }

    /// A camera translated along x by `shift`, R = I, same intrinsics as `makeTestCamera`.
    private func camera(width: Int, height: Int, shift: Float) -> Camera {
        Camera(width: width, height: height,
               fx: Float(height), fy: Float(height),
               cx: Float(width) / 2, cy: Float(height) / 2,
               rotation: .identity, translation: SIMD3(shift, 0, 0))
    }

    private func makeRenderTarget(_ context: MetalContext, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        return texture
    }

    private func makeImageGradientBuffer(_ context: MetalContext, _ values: [SIMD3<Float>]) throws -> MTLBuffer {
        let buffer = try context.makeBuffer(length: values.count * 4 * MemoryLayout<Float>.size)
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count * 4)
        for (i, value) in values.enumerated() {
            pointer[4 * i] = value.x
            pointer[4 * i + 1] = value.y
            pointer[4 * i + 2] = value.z
            pointer[4 * i + 3] = 0
        }
        return buffer
    }

    private func randomImageGradient(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            let x = Float.random(in: -1...1, using: &rng)
            let y = Float.random(in: -1...1, using: &rng)
            let z = Float.random(in: -1...1, using: &rng)
            values.append(SIMD3(x, y, z))
        }
        return values
    }

    /// Writes CPU statistics into the GPU buffers (CPU access while the GPU is idle).
    private func uploadStatistics(_ statistics: DensificationStatistics, to buffers: DensificationStatisticsBuffers) {
        let n = statistics.gradientNormSums.count
        guard n > 0 else { return }
        let sums = buffers.gradientNormSums.contents().bindMemory(to: Float.self, capacity: n)
        let counts = buffers.visibleCounts.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            sums[i] = statistics.gradientNormSums[i]
            counts[i] = statistics.visibleCounts[i]
        }
    }

    private func expectClose(_ actual: [Float], _ expected: [Float], relative: Float = 1e-5,
                             absolute: Float = 1e-5, label: String) {
        #expect(actual.count == expected.count, "\(label): length \(actual.count) vs \(expected.count)")
        guard actual.count == expected.count else { return }
        var failures = 0
        for (a, e) in zip(actual, expected) where !(abs(a - e) <= absolute + relative * abs(e)) {
            failures += 1
        }
        #expect(failures == 0, "\(label): \(failures) values differ")
    }

    private func expectSameCloud(_ actual: GaussianCloud, _ expected: GaussianCloud, label: String) {
        #expect(actual.count == expected.count, "\(label): count")
        #expect(actual.shDegree == expected.shDegree, "\(label): SH degree")
        expectClose(actual.positions, expected.positions, label: "\(label) positions")
        expectClose(actual.logScales, expected.logScales, label: "\(label) logScales")
        expectClose(actual.rotations, expected.rotations, label: "\(label) rotations")
        expectClose(actual.opacityLogits, expected.opacityLogits, label: "\(label) opacity")
        expectClose(actual.sh, expected.sh, label: "\(label) sh")
    }

    private func expectSameMoments(_ actual: GaussianMoments, _ expected: GaussianMoments, label: String) {
        expectClose(actual.positions, expected.positions, label: "\(label) positions")
        expectClose(actual.logScales, expected.logScales, label: "\(label) logScales")
        expectClose(actual.rotations, expected.rotations, label: "\(label) rotations")
        expectClose(actual.opacityLogits, expected.opacityLogits, label: "\(label) opacity")
        expectClose(actual.sh, expected.sh, label: "\(label) sh")
    }

    /// Moments with a distinct value per float (one running index over all groups), so permuted,
    /// copied or wrongly zeroed moments change the comparison.
    private func distinctMoments(like cloud: GaussianCloud, scale: Float, offset: Float) -> GaussianMoments {
        var next = 0
        func fill(_ count: Int) -> [Float] {
            var values = [Float](repeating: 0, count: count)
            for i in 0..<count {
                values[i] = Float(next) * scale + offset
                next += 1
            }
            return values
        }
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        let positions = fill(3 * cloud.count)
        let logScales = fill(3 * cloud.count)
        let rotations = fill(4 * cloud.count)
        let opacity = fill(cloud.count)
        let sh = fill(cloud.count * k * 3)
        return GaussianMoments(positions: positions, logScales: logScales, rotations: rotations,
                               opacityLogits: opacity, sh: sh)
    }

    /// Adam state with distinct, non-zero moments: m = 1 + 1e-3·i, v = 2 + 2e-3·i.
    private func makeState(_ cloud: GaussianCloud) -> AdamState {
        var state = AdamState(zeroLike: cloud)
        state.m = distinctMoments(like: cloud, scale: 1e-3, offset: 1)
        state.v = distinctMoments(like: cloud, scale: 2e-3, offset: 2)
        return state
    }

    private func withOpacity(_ cloud: GaussianCloud, _ logits: [Float]) throws -> GaussianCloud {
        try GaussianCloud(shDegree: cloud.shDegree, positions: cloud.positions, logScales: cloud.logScales,
                          rotations: cloud.rotations, opacityLogits: logits, sh: cloud.sh)
    }

    /// Clone-sized Gaussians (all scales below the clone threshold for scene radius 1).
    private func smallCloud(count: Int, seed: UInt64, shDegree: Int = 0) throws -> GaussianCloud {
        try makeRandomCloud(count: count, seed: seed, shDegree: shDegree, scale: 0.002...0.008)
    }

    /// Runs the CPU densification and returns the plan, the new cloud and the new moments.
    private func cpuDensify(_ cloud: GaussianCloud, statistics: DensificationStatistics, state: AdamState,
                            policy: DensificationPolicy, sceneRadius: Float, iteration: Int, seed: UInt32,
                            maxGaussians: Int) -> (plan: DensifyPlan, cloud: GaussianCloud, state: AdamState) {
        let plan = Densification.plan(cloud, statistics: statistics, policy: policy, sceneRadius: sceneRadius,
                                      iteration: iteration, maxGaussians: maxGaussians)
        var updated = state
        let result = Densification.apply(plan, to: cloud, state: &updated, policy: policy,
                                         seed: seed, iteration: iteration)
        return (plan, result, updated)
    }

    /// Uploads the inputs, runs `GPUDensifier.densify` once and returns the buffers and the summary.
    private func runGPU(context: MetalContext, cloud: GaussianCloud, state: AdamState,
                        statistics: DensificationStatistics, policy: DensificationPolicy,
                        sceneRadius: Float, iteration: Int, seed: UInt32, maxGaussians: Int) throws -> GPURun {
        let store = try GaussianBuffers(context: context, cloud: cloud)
        let m = try GaussianBuffers(context: context, capacity: cloud.count, shDegree: cloud.shDegree)
        m.upload(arrays: state.m)
        let v = try GaussianBuffers(context: context, capacity: cloud.count, shDegree: cloud.shDegree)
        v.upload(arrays: state.v)
        let gpuStatistics = try DensificationStatisticsBuffers(context: context, capacity: cloud.count)
        uploadStatistics(statistics, to: gpuStatistics)
        let densifier = try GPUDensifier(context: context)
        let summary = try densifier.densify(parameters: store, m: m, v: v, statistics: gpuStatistics,
                                            policy: policy, sceneRadius: sceneRadius, iteration: iteration,
                                            seed: seed, maxGaussians: maxGaussians)
        return GPURun(store: store, m: m, v: v, statistics: gpuStatistics, summary: summary)
    }

    // MARK: - Statistics

    @Test func statisticsMatchCPU() throws {
        let context = try makeTestContext()
        let width = 64
        let height = 48
        let views = 3
        let cloud = try makeRandomCloud(count: 300, seed: 7, shDegree: 0)
        let store = try GaussianBuffers(context: context, cloud: cloud)
        let renderer = try SplatRenderer(context: context, gaussians: store)
        let statistics = try DensificationStatisticsBuffers(context: context, capacity: cloud.count)
        var cpuStatistics = DensificationStatistics(count: cloud.count)
        let target = try makeRenderTarget(context, width: width, height: height)

        guard let reset = context.queue.makeCommandBuffer() else { throw MetalError.commandFailed("no command buffer") }
        try statistics.encodeReset(commandBuffer: reset)
        reset.commit()
        reset.waitUntilCompleted()

        for view in 0..<views {
            let cam = camera(width: width, height: height, shift: 0.15 * Float(view))
            let pass = try renderer.encodeForward(camera: cam, settings: settings(), to: target)
            let gradientValues = randomImageGradient(count: width * height, seed: UInt64(view + 1))
            let gradientBuffer = try makeImageGradientBuffer(context, gradientValues)
            let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: cloud.shDegree)
            try renderer.encodeBackward(imageGradient: gradientBuffer, into: gradients,
                                        statistics: statistics, commandBuffer: pass.commandBuffer)
            pass.commandBuffer.commit()
            pass.commandBuffer.waitUntilCompleted()
            #expect(pass.commandBuffer.status != .error)

            let cpu = CPURasterizer.backwardWithScreenSpace(cloud, camera: cam, settings: settings(),
                                                            imageGradient: gradientValues)
            cpuStatistics.accumulate(mean2D: cpu.mean2D, visible: cpu.visible, width: width, height: height)
        }

        let gpu = statistics.read(count: cloud.count)
        #expect(gpu.visibleCounts == cpuStatistics.visibleCounts, "visible counts must match exactly")
        // Sums of `views` terms, each scaled by W/2 (pixels); Phase 2 tolerance per term.
        let scale: Float = Float(views) * (Float(width) / 2)
        var failures = 0
        for i in 0..<cloud.count {
            let expected = cpuStatistics.gradientNormSums[i]
            let actual = gpu.gradientNormSums[i]
            let bound: Float = scale * 1e-4 + 1e-3 * abs(expected)
            if !(abs(actual - expected) <= bound) {
                failures += 1
            }
        }
        #expect(failures == 0, "gradient norm sums differ for \(failures) Gaussians")
    }

    /// Exact check of the statistics kernel: the expected values are computed on the CPU from the very
    /// accumulator the kernel read (slots 7-8 after rasterizeBackward), so only the norm arithmetic is compared.
    @Test func statisticsKernelMatchesCPUOnAccumulator() throws {
        let context = try makeTestContext()
        let width = 64
        let height = 48
        let cloud = try makeRandomCloud(count: 300, seed: 17, shDegree: 0)
        let store = try GaussianBuffers(context: context, cloud: cloud)
        let renderer = try SplatRenderer(context: context, gaussians: store)
        let statistics = try DensificationStatisticsBuffers(context: context, capacity: cloud.count)
        let target = try makeRenderTarget(context, width: width, height: height)
        let cam = camera(width: width, height: height, shift: 0.1)

        let pass = try renderer.encodeForward(camera: cam, settings: settings(), to: target)
        let gradientBuffer = try makeImageGradientBuffer(context, randomImageGradient(count: width * height, seed: 3))
        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: 0)
        try renderer.encodeBackward(imageGradient: gradientBuffer, into: gradients,
                                    statistics: statistics, commandBuffer: pass.commandBuffer)
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        #expect(pass.commandBuffer.status != .error)

        guard let accumulatorBuffer = renderer.gradientAccumulator, let touchedBuffer = renderer.tilesTouched else {
            throw MetalError.commandFailed("forward buffers are missing")
        }
        let n = cloud.count
        let accumulator = accumulatorBuffer.contents().bindMemory(to: Float.self, capacity: n * 12)
        let touched = touchedBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let gpu = statistics.read(count: n)
        let halfWidth: Float = Float(width) / 2
        let halfHeight: Float = Float(height) / 2

        var visibleCount = 0
        var countFailures = 0
        var sumFailures = 0
        for i in 0..<n {
            let isVisible = touched[i] > 0
            let expectedCount: UInt32 = isVisible ? 1 : 0
            if gpu.visibleCounts[i] != expectedCount {
                countFailures += 1
            }
            var expectedSum: Float = 0
            if isVisible {
                visibleCount += 1
                let sx: Float = accumulator[12 * i + 7] * halfWidth
                let sy: Float = accumulator[12 * i + 8] * halfHeight
                expectedSum = (sx * sx + sy * sy).squareRoot()
            }
            let actual = gpu.gradientNormSums[i]
            if !(abs(actual - expectedSum) <= 1e-6 * max(1, abs(expectedSum))) {
                sumFailures += 1
            }
        }
        #expect(visibleCount > 0, "the view must see some Gaussians")
        #expect(countFailures == 0, "visible counts differ for \(countFailures) Gaussians")
        #expect(sumFailures == 0, "gradient norm sums differ for \(sumFailures) Gaussians")
    }

    @Test func backwardWithoutStatisticsIsUnchanged() throws {
        let context = try makeTestContext()
        let width = 64
        let height = 48
        let cloud = try makeRandomCloud(count: 200, seed: 11, shDegree: 0)
        let store = try GaussianBuffers(context: context, cloud: cloud)
        let renderer = try SplatRenderer(context: context, gaussians: store)
        let statistics = try DensificationStatisticsBuffers(context: context, capacity: cloud.count)
        let target = try makeRenderTarget(context, width: width, height: height)
        let cam = camera(width: width, height: height, shift: 0)
        let gradientBuffer = try makeImageGradientBuffer(context, randomImageGradient(count: width * height, seed: 5))

        func gradients(withStatistics hook: DensificationStatisticsBuffers?) throws -> GaussianGradients {
            let pass = try renderer.encodeForward(camera: cam, settings: settings(), to: target)
            let buffers = try GradientBuffers(context: context, count: cloud.count, shDegree: 0)
            try renderer.encodeBackward(imageGradient: gradientBuffer, into: buffers,
                                        statistics: hook, commandBuffer: pass.commandBuffer)
            pass.commandBuffer.commit()
            pass.commandBuffer.waitUntilCompleted()
            return buffers.toGaussianGradients(count: cloud.count, shDegree: 0)
        }

        // Float atomics make two passes differ in the last bits, so compare with the Phase 2 tolerance.
        let plain = try gradients(withStatistics: nil)
        let hooked = try gradients(withStatistics: statistics)
        expectClose(plain.positions, hooked.positions, relative: 1e-5, absolute: 1e-6, label: "positions")
        expectClose(plain.logScales, hooked.logScales, relative: 1e-5, absolute: 1e-6, label: "logScales")
        expectClose(plain.rotations, hooked.rotations, relative: 1e-5, absolute: 1e-6, label: "rotations")
        expectClose(plain.opacityLogits, hooked.opacityLogits, relative: 1e-5, absolute: 1e-6, label: "opacity")
        expectClose(plain.sh, hooked.sh, relative: 1e-5, absolute: 1e-6, label: "sh")
    }

    // MARK: - Decisions and thresholds

    /// The scale prune is strictly after `pruneScaleStartIteration`; the GPU side is covered by
    /// `decisionsAndLayoutMatchCPU` running at `pruneScaleStartIteration + 1`.
    @Test func scalePruneStartsAfterPruneScaleStartIteration() {
        let policy = DensificationPolicy()
        let start = policy.pruneScaleStartIteration
        let unitScale = SIMD3<Float>(repeating: 0)   // max scale exp(0) = 1
        let sceneRadius: Float = 2                   // prune threshold 0.1 · 2 = 0.2

        let atStart = Densification.action(logScale: unitScale, opacityLogit: 0, gradientNormSum: 0,
                                           visibleCount: 0, policy: policy, sceneRadius: sceneRadius,
                                           iteration: start)
        let afterStart = Densification.action(logScale: unitScale, opacityLogit: 0, gradientNormSum: 0,
                                              visibleCount: 0, policy: policy, sceneRadius: sceneRadius,
                                              iteration: start + 1)
        #expect(atStart == .keep)
        #expect(afterStart == .prune)
    }

    @Test func decisionsAndLayoutMatchCPU() throws {
        let context = try makeTestContext()
        let policy = DensificationPolicy()
        let sceneRadius: Float = 2
        // Scale pruning is active only after pruneScaleStartIteration; run just past the boundary.
        let iteration = policy.pruneScaleStartIteration + 1
        let seed: UInt32 = 0x1234
        let maxGaussians = 1_000_000
        let base = try makeRandomCloud(count: 300, seed: 21, shDegree: 1, scale: 0.005...0.3)
        // Opacity-prune branch: every 37th Gaussian gets opacity sigmoid(-8) < pruneOpacity.
        let opacityPruned = Array(stride(from: 0, to: base.count, by: 37))
        var logits = base.opacityLogits
        for i in opacityPruned {
            logits[i] = -8
        }
        let cloud = try withOpacity(base, logits)

        // Crafted statistics: mixed visible counts and average gradients around the threshold.
        var rng = SplitMix64(seed: 99)
        var statistics = DensificationStatistics(count: cloud.count)
        for i in 0..<cloud.count {
            statistics.visibleCounts[i] = UInt32.random(in: 0...3, using: &rng)
            statistics.gradientNormSums[i] = Float.random(in: 0...0.001, using: &rng)
        }

        let state = makeState(cloud)
        let cpu = cpuDensify(cloud, statistics: statistics, state: state, policy: policy,
                             sceneRadius: sceneRadius, iteration: iteration, seed: seed, maxGaussians: maxGaussians)
        let plan = cpu.plan

        let gpu = try runGPU(context: context, cloud: cloud, state: state, statistics: statistics, policy: policy,
                             sceneRadius: sceneRadius, iteration: iteration, seed: seed, maxGaussians: maxGaussians)
        let summary = gpu.summary

        #expect(summary.after == plan.outputCount)
        #expect(summary.pruned == plan.pruned)
        #expect(summary.cloned == plan.cloned)
        #expect(summary.split == plan.split)
        #expect(summary.demoted == plan.demoted)
        #expect(summary.keptAll == plan.keptAll)
        expectSameCloud(gpu.store.snapshot(), cpu.cloud, label: "parameters")
        expectSameMoments(gpu.m.snapshotArrays(), cpu.state.m, label: "m")
        expectSameMoments(gpu.v.snapshotArrays(), cpu.state.v, label: "v")

        // The fixture must exercise every branch to be meaningful.
        #expect(plan.pruned > 0 && plan.cloned > 0 && plan.split > 0)
        #expect(opacityPruned.allSatisfy { plan.actions[$0] == .prune }, "opacity prune branch not hit")
        #expect(gpu.statistics.read(count: gpu.store.count) == DensificationStatistics(count: gpu.store.count))
    }

    // MARK: - Budget and capacity

    @Test func densifyRespectsCap() throws {
        let context = try makeTestContext()
        let policy = DensificationPolicy()
        let cloud = try smallCloud(count: 100, seed: 3)
        // Growth candidates at non-contiguous indices 7, 9, ..., 85 (40 of them), no prunes.
        // With maxGaussians = count + 5 only the first five candidates in index order may grow.
        let candidates = (0..<40).map { 7 + 2 * $0 }
        var statistics = DensificationStatistics(count: cloud.count)
        for i in candidates {
            statistics.visibleCounts[i] = 1
            statistics.gradientNormSums[i] = 1
        }
        let maxGaussians = cloud.count + 5
        let state = makeState(cloud)
        let cpu = cpuDensify(cloud, statistics: statistics, state: state, policy: policy,
                             sceneRadius: 1, iteration: 1_000, seed: 1, maxGaussians: maxGaussians)

        #expect(cpu.plan.pruned == 0)
        #expect(cpu.plan.demoted == 35)
        #expect(cpu.plan.demoted > 0)
        #expect(cpu.plan.outputCount == maxGaussians)
        #expect(candidates.prefix(5).allSatisfy { cpu.plan.actions[$0] == .clone })
        #expect(candidates.dropFirst(5).allSatisfy { cpu.plan.actions[$0] == .keep })

        let gpu = try runGPU(context: context, cloud: cloud, state: state, statistics: statistics, policy: policy,
                             sceneRadius: 1, iteration: 1_000, seed: 1, maxGaussians: maxGaussians)
        let summary = gpu.summary
        #expect(summary.pruned == 0)
        #expect(summary.demoted == 35)
        #expect(summary.after == cpu.plan.outputCount)
        #expect(summary.after <= maxGaussians)
        #expect(gpu.store.count == summary.after)
        #expect(gpu.store.count <= gpu.store.capacity)
        expectSameCloud(gpu.store.snapshot(), cpu.cloud, label: "parameters")
        expectSameMoments(gpu.m.snapshotArrays(), cpu.state.m, label: "m")
        expectSameMoments(gpu.v.snapshotArrays(), cpu.state.v, label: "v")
        #expect(gpu.statistics.read(count: gpu.store.count) == DensificationStatistics(count: gpu.store.count))
    }

    @Test func densifyGrowsCapacity() throws {
        let context = try makeTestContext()
        let policy = DensificationPolicy()
        let cloud = try smallCloud(count: 60, seed: 4)
        var statistics = DensificationStatistics(count: cloud.count)
        for i in 0..<50 {
            statistics.visibleCounts[i] = 1
            statistics.gradientNormSums[i] = 1
        }
        let state = makeState(cloud)
        let cpu = cpuDensify(cloud, statistics: statistics, state: state, policy: policy,
                             sceneRadius: 1, iteration: 1_000, seed: 2, maxGaussians: 1_000_000)

        // Capacity equals the count, so 50 clones force a reallocation.
        let gpu = try runGPU(context: context, cloud: cloud, state: state, statistics: statistics, policy: policy,
                             sceneRadius: 1, iteration: 1_000, seed: 2, maxGaussians: 1_000_000)
        let summary = gpu.summary

        #expect(summary.grewCapacity)
        #expect(summary.after == 110)
        #expect(summary.after == cpu.plan.outputCount)
        #expect(gpu.store.capacity >= summary.after)
        #expect(gpu.store.count == 110)
        expectSameCloud(gpu.store.snapshot(), cpu.cloud, label: "parameters")
        expectSameMoments(gpu.m.snapshotArrays(), cpu.state.m, label: "m")
        expectSameMoments(gpu.v.snapshotArrays(), cpu.state.v, label: "v")
    }

    /// Statistics must cover the new count even when the parameters did not grow: capacity 1000 with
    /// count 100 and statistics of capacity 100, after 50 clones (count 150).
    @Test func statisticsCoverCountWithoutParameterGrowth() throws {
        let context = try makeTestContext()
        let policy = DensificationPolicy()
        let cloud = try smallCloud(count: 100, seed: 8)
        let state = makeState(cloud)
        let store = try GaussianBuffers(context: context, capacity: 1_000, shDegree: 0)
        store.upload(cloud)
        let m = try GaussianBuffers(context: context, capacity: 1_000, shDegree: 0)
        m.upload(arrays: state.m)
        let v = try GaussianBuffers(context: context, capacity: 1_000, shDegree: 0)
        v.upload(arrays: state.v)
        let statistics = try DensificationStatisticsBuffers(context: context, capacity: cloud.count)
        var cpuStatistics = DensificationStatistics(count: cloud.count)
        for i in 0..<50 {
            cpuStatistics.visibleCounts[i] = 1
            cpuStatistics.gradientNormSums[i] = 1
        }
        uploadStatistics(cpuStatistics, to: statistics)

        let densifier = try GPUDensifier(context: context)
        let summary = try densifier.densify(parameters: store, m: m, v: v, statistics: statistics, policy: policy,
                                            sceneRadius: 1, iteration: 1_000, seed: 8, maxGaussians: 1_000_000)

        #expect(!summary.grewCapacity)
        #expect(summary.after == 150)
        #expect(store.count == 150)
        #expect(statistics.capacity >= summary.after)
        // The statistics must now be usable for the new count.
        #expect(statistics.read(count: store.count) == DensificationStatistics(count: store.count))
    }

    @Test func neverPrunesEverything() throws {
        let context = try makeTestContext()
        let policy = DensificationPolicy()
        let base = try makeRandomCloud(count: 50, seed: 5, shDegree: 0)
        // Opacity logit -20 for every Gaussian: all would be pruned.
        let cloud = try withOpacity(base, [Float](repeating: -20, count: base.count))
        let statistics = DensificationStatistics(count: cloud.count)
        let state = makeState(cloud)
        let cpu = cpuDensify(cloud, statistics: statistics, state: state, policy: policy,
                             sceneRadius: 1, iteration: 1_000, seed: 3, maxGaussians: 1_000_000)
        #expect(cpu.plan.keptAll)

        let gpu = try runGPU(context: context, cloud: cloud, state: state, statistics: statistics, policy: policy,
                             sceneRadius: 1, iteration: 1_000, seed: 3, maxGaussians: 1_000_000)
        let summary = gpu.summary

        #expect(summary.keptAll)
        #expect(summary.after == cloud.count)
        #expect(summary.pruned == 0)
        #expect(gpu.store.count == cloud.count)
        expectSameCloud(gpu.store.snapshot(), cpu.cloud, label: "parameters")
    }
}
