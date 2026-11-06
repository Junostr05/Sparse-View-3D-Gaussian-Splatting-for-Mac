import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU artifact regularizers (anisotropy gradient, scale cap) against the CPU references in SplatCore,
/// the anisotropy golden and finite differences, plus CPU compaction of the parameter and Adam buffers.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPURegularizerTests {

    // MARK: - Helpers

    private func makeCommandBuffer(_ context: MetalContext) throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        return commandBuffer
    }

    /// Commits the buffer and blocks until the GPU finished it.
    private func runAndWait(_ commandBuffer: MTLCommandBuffer) {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil, "command buffer failed: \(String(describing: commandBuffer.error))")
    }

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func randomFloats(_ n: Int, _ rng: inout SplitMix64) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(n)
        for _ in 0..<n {
            values.append(Float.random(in: -1...1, using: &rng))
        }
        return values
    }

    private func randomGradients(count: Int, shDegree: Int, rng: inout SplitMix64) -> GaussianGradients {
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        return GaussianGradients(
            positions: randomFloats(3 * count, &rng),
            logScales: randomFloats(3 * count, &rng),
            rotations: randomFloats(4 * count, &rng),
            opacityLogits: randomFloats(count, &rng),
            sh: randomFloats(count * k * 3, &rng))
    }

    /// Gradient buffers with every component zeroed (the regularizer adds into `logScales`).
    private func makeZeroGradients(_ context: MetalContext, count: Int, shDegree: Int) throws -> GradientBuffers {
        let gradients = try GradientBuffers(context: context, count: count, shDegree: shDegree)
        let buffers: [MTLBuffer] = [gradients.positions, gradients.logScales, gradients.rotations,
                                    gradients.opacityLogits, gradients.sh]
        for buffer in buffers {
            _ = buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        return gradients
    }

    /// Degree-0 cloud from flat arrays with identity rotations and zero opacity and colour.
    private func makeFlatCloud(positions: [Float], logScales: [Float]) throws -> GaussianCloud {
        let n = positions.count / 3
        var rotations: [Float] = []
        rotations.reserveCapacity(4 * n)
        for _ in 0..<n {
            rotations += [1, 0, 0, 0]
        }
        return try GaussianCloud(shDegree: 0,
                                 positions: positions,
                                 logScales: logScales,
                                 rotations: rotations,
                                 opacityLogits: [Float](repeating: 0, count: n),
                                 sh: [Float](repeating: 0, count: n * 3))
    }

    private func cpuAnisotropy(_ cloud: GaussianCloud, weight: Float, maxRatio: Float)
        -> (loss: Double, gradLogScales: [Float]) {
        var gradients = GaussianGradients(zeroLike: cloud)
        let loss: Double = Regularizers.addAnisotropyGradient(cloud, weight: weight,
                                                              maxRatio: maxRatio, into: &gradients)
        return (loss, gradients.logScales)
    }

    /// Runs `GPURegularizers.encodeAnisotropy` on a fresh store and returns the gradient `logScales`.
    private func gpuAnisotropy(_ context: MetalContext, cloud: GaussianCloud,
                               weight: Float, maxRatio: Float) throws -> [Float] {
        let parameters = try GaussianBuffers(context: context, cloud: cloud)
        let gradients = try makeZeroGradients(context, count: cloud.count, shDegree: cloud.shDegree)
        let regularizers = try GPURegularizers(context: context)
        let commandBuffer = try makeCommandBuffer(context)
        try regularizers.encodeAnisotropy(parameters: parameters, gradients: gradients,
                                          weight: weight, maxRatio: maxRatio, commandBuffer: commandBuffer)
        runAndWait(commandBuffer)
        return readFloats(gradients.logScales, count: 3 * cloud.count)
    }

    private func maxMagnitude(_ values: [Float]) -> Float {
        var result: Float = 0
        for value in values {
            result = max(result, abs(value))
        }
        return result
    }

    /// Keeps the Gaussians whose flag is true, preserving order, for one flat per-Gaussian group.
    private func filterGroup(_ values: [Float], stride: Int, keep: [Bool]) -> [Float] {
        var out: [Float] = []
        for i in 0..<keep.count where keep[i] {
            out.append(contentsOf: values[(i * stride)..<((i + 1) * stride)])
        }
        return out
    }

    private func filter(_ arrays: GaussianGradients, keep: [Bool], shDegree: Int) -> GaussianGradients {
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        return GaussianGradients(
            positions: filterGroup(arrays.positions, stride: 3, keep: keep),
            logScales: filterGroup(arrays.logScales, stride: 3, keep: keep),
            rotations: filterGroup(arrays.rotations, stride: 4, keep: keep),
            opacityLogits: filterGroup(arrays.opacityLogits, stride: 1, keep: keep),
            sh: filterGroup(arrays.sh, stride: 3 * k, keep: keep))
    }

    private func arrays(of cloud: GaussianCloud) -> GaussianGradients {
        GaussianGradients(positions: cloud.positions, logScales: cloud.logScales,
                          rotations: cloud.rotations, opacityLogits: cloud.opacityLogits, sh: cloud.sh)
    }

    private func positionsOf(_ flat: [Float]) -> [SIMD3<Float>] {
        stride(from: 0, to: flat.count, by: 3).map { SIMD3<Float>(flat[$0], flat[$0 + 1], flat[$0 + 2]) }
    }

    // MARK: - Anisotropy

    @Test func anisotropyMatchesCPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 1_000, seed: 7, shDegree: 1, scale: 0.01...1.0)
        let weight: Float = 0.1
        let maxRatio: Float = 10
        let cpu = cpuAnisotropy(cloud, weight: weight, maxRatio: maxRatio).gradLogScales
        let gpu = try gpuAnisotropy(context, cloud: cloud, weight: weight, maxRatio: maxRatio)

        let scale = maxMagnitude(cpu)
        #expect(scale > 0, "the random cloud should activate the hinge")
        #expect(gpu.count == cpu.count)
        var worst: Float = 0
        for (a, b) in zip(gpu, cpu) {
            worst = max(worst, abs(a - b))
        }
        print("anisotropy GPU vs CPU: max abs error \(worst), max |grad| \(scale), relative \(worst / scale)")
        #expect(worst <= 1e-5 * scale)
    }

    @Test func anisotropyMatchesGolden() throws {
        let context = try makeTestContext()
        let golden = try AnisotropyGolden.load("anisotropy")
        let cloud = try makeFlatCloud(positions: [Float](repeating: 0, count: 3 * golden.count),
                                      logScales: golden.logScales)
        let gpu = try gpuAnisotropy(context, cloud: cloud, weight: golden.weight, maxRatio: golden.maxRatio)

        #expect(gpu.count == golden.gradLogScales.count)
        var worst: Float = 0
        for (a, b) in zip(gpu, golden.gradLogScales) {
            worst = max(worst, abs(a - b))
            #expect(abs(a - b) <= 1e-5 * abs(b) + 1e-7)
        }
        print("anisotropy GPU vs golden: max abs error \(worst)")
        // The GPU kernel writes gradients only; the loss value is checked on the CPU reference.
        let loss = cpuAnisotropy(cloud, weight: golden.weight, maxRatio: golden.maxRatio).loss
        #expect(abs(loss - golden.loss) <= 1e-6)
    }

    @Test func anisotropyMatchesFiniteDifferences() throws {
        let context = try makeTestContext()
        let base = try makeRandomCloud(count: 8, seed: 3, shDegree: 0, scale: 0.005...1.0)
        let weight: Float = 0.1
        let maxRatio: Float = 10
        let analytic = try gpuAnisotropy(context, cloud: base, weight: weight, maxRatio: maxRatio)

        var activeGaussians = 0
        for i in 0..<base.count where analytic[3 * i] != 0 || analytic[3 * i + 1] != 0 || analytic[3 * i + 2] != 0 {
            activeGaussians += 1
        }
        #expect(activeGaussians >= 3, "the finite-difference cloud should contain active Gaussians")

        let basePositions = base.positions
        let baseLogs = base.logScales
        let h: Double = 1e-2
        var worst: Double = 0
        for flat in 0..<analytic.count {
            let x0 = Double(baseLogs[flat])
            // Use the steps actually representable in Float, so the quotient divides by the true step.
            let up = Float(x0 + h)
            let down = Float(x0 - h)
            var plus = baseLogs
            var minus = baseLogs
            plus[flat] = up
            minus[flat] = down
            let lossPlus = cpuAnisotropy(try makeFlatCloud(positions: basePositions, logScales: plus),
                                         weight: weight, maxRatio: maxRatio).loss
            let lossMinus = cpuAnisotropy(try makeFlatCloud(positions: basePositions, logScales: minus),
                                          weight: weight, maxRatio: maxRatio).loss
            let fd = (lossPlus - lossMinus) / (Double(up) - Double(down))
            let error = abs(Double(analytic[flat]) - fd)
            worst = max(worst, error / max(abs(fd), 1e-3))
            #expect(error <= 1e-4 * max(abs(fd), 1e-3), "entry \(flat): gpu \(analytic[flat]) vs fd \(fd)")
        }
        print("anisotropy GPU vs finite differences: worst relative error \(worst)")
    }

    // MARK: - Scale cap

    @Test func scaleCapMatchesCPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 1_000, seed: 5, shDegree: 1, depth: 2...8, scale: 0.01...1.0)
        let extent = SceneExtent(center: SIMD3<Float>(0, 0, 5), radius: 3)
        let fraction: Float = 0.05

        var cpuCloud = cloud
        Regularizers.applyScaleCap(&cpuCloud, fraction: fraction, extent: extent)

        let parameters = try GaussianBuffers(context: context, cloud: cloud)
        let regularizers = try GPURegularizers(context: context)
        let commandBuffer = try makeCommandBuffer(context)
        try regularizers.encodeScaleCap(parameters: parameters, fraction: fraction, extent: extent,
                                        commandBuffer: commandBuffer)
        runAndWait(commandBuffer)

        let gpu = readFloats(parameters.logScales, count: 3 * cloud.count)
        let cpu = cpuCloud.logScales
        var worst: Float = 0
        var changed = 0
        for i in 0..<gpu.count {
            worst = max(worst, abs(gpu[i] - cpu[i]))
            if cloud.logScales[i] != cpu[i] {
                changed += 1
            }
        }
        print("scale cap GPU vs CPU: max abs error \(worst), entries capped \(changed)")
        #expect(changed > 0, "the cap should be active on this cloud")
        #expect(worst <= 1e-5)
    }

    // MARK: - Compaction

    @Test func compactPreservesOrderAndCount() throws {
        let context = try makeTestContext()
        let n = 300
        let cloud = try makeRandomCloud(count: n, seed: 9, shDegree: 1)
        let store = try GaussianBuffers(context: context, cloud: cloud)

        var rng = SplitMix64(seed: 21)
        let momentsM = randomGradients(count: n, shDegree: 1, rng: &rng)
        let momentsV = randomGradients(count: n, shDegree: 1, rng: &rng)
        let m = try GaussianBuffers(context: context, capacity: n, shDegree: 1)
        let v = try GaussianBuffers(context: context, capacity: n, shDegree: 1)
        m.upload(arrays: momentsM)
        v.upload(arrays: momentsV)

        let keep: [Bool] = (0..<n).map { $0 % 3 == 0 }
        let keptCount = keep.filter { $0 }.count
        let expectedCloud = filter(arrays(of: cloud), keep: keep, shDegree: 1)
        let expectedM = filter(momentsM, keep: keep, shDegree: 1)
        let expectedV = filter(momentsV, keep: keep, shDegree: 1)

        store.compact(keeping: keep)
        m.compact(keeping: keep)
        v.compact(keeping: keep)

        #expect(store.count == keptCount)
        #expect(m.count == keptCount)
        #expect(v.count == keptCount)
        #expect(store.snapshotArrays() == expectedCloud)
        #expect(m.snapshotArrays() == expectedM)
        #expect(v.snapshotArrays() == expectedV)
        #expect(store.readPositions() == positionsOf(expectedCloud.positions))
    }

    @Test func compactNeverReadsBeyondCount() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 10, seed: 13, shDegree: 0)
        let capacity = 64
        let store = try GaussianBuffers(context: context, capacity: capacity, shDegree: 0)
        store.upload(cloud)

        // Poison every spare slot so that any read past `count` would surface as NaN.
        let floats = GaussianBuffers.floatsPerGaussian(shDegree: 0)
        for (group, buffer) in store.groups.enumerated() {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: capacity * floats[group])
            for i in (10 * floats[group])..<(capacity * floats[group]) {
                pointer[i] = Float.nan
            }
        }

        let initial = store.readPositions()
        #expect(initial.count == 10)
        #expect(initial == positionsOf(cloud.positions))

        let keep: [Bool] = (0..<10).map { $0 % 2 == 0 }
        store.compact(keeping: keep)
        #expect(store.count == 5)
        let after = store.readPositions()
        #expect(after.count == 5)
        #expect(after.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
        #expect(after == positionsOf(filterGroup(cloud.positions, stride: 3, keep: keep)))

        // Slots at and beyond the old count were not written by the compaction.
        let positionPointer = store.positions.contents().bindMemory(to: Float.self, capacity: capacity * 3)
        #expect(positionPointer[30].isNaN)
        #expect(positionPointer[capacity * 3 - 1].isNaN)
    }
}
