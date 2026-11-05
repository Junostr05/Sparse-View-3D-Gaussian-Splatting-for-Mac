import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Metal Adam against the CPU reference `CPUAdam` (SplatCore).
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPUAdamTests {

    // MARK: - Helpers

    private func makeRates() -> ResolvedLearningRates {
        ResolvedLearningRates(position: 1.6e-4, logScale: 5e-3, rotation: 1e-3,
                              opacity: 5e-2, shDC: 2.5e-3, shRest: 1.25e-4)
    }

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

    private func randomFloats(_ n: Int, _ rng: inout SplitMix64) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(n)
        for _ in 0..<n {
            values.append(Float.random(in: -1...1, using: &rng))
        }
        return values
    }

    private func makeRandomGradients(count: Int, shDegree: Int, rng: inout SplitMix64) -> GaussianGradients {
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        return GaussianGradients(
            positions: randomFloats(3 * count, &rng),
            logScales: randomFloats(3 * count, &rng),
            rotations: randomFloats(4 * count, &rng),
            opacityLogits: randomFloats(count, &rng),
            sh: randomFloats(count * k * 3, &rng))
    }

    private func writeFloats(_ values: [Float], to buffer: MTLBuffer) {
        guard !values.isEmpty else { return }
        values.withUnsafeBytes { raw in
            buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
    }

    private func writeGradients(_ gradients: GaussianGradients, to buffers: GradientBuffers) {
        writeFloats(gradients.positions, to: buffers.positions)
        writeFloats(gradients.logScales, to: buffers.logScales)
        writeFloats(gradients.rotations, to: buffers.rotations)
        writeFloats(gradients.opacityLogits, to: buffers.opacityLogits)
        writeFloats(gradients.sh, to: buffers.sh)
    }

    /// Sets floats `start..<end` of `buffer` to `value`.
    private func fillFloats(_ buffer: MTLBuffer, from start: Int, to end: Int, value: Float) {
        guard end > start else { return }
        for i in start..<end {
            buffer.contents().storeBytes(of: value, toByteOffset: i * MemoryLayout<Float>.size, as: Float.self)
        }
    }

    /// Raw bit patterns of floats `start..<end` of `buffer`.
    private func floatBits(_ buffer: MTLBuffer, from start: Int, to end: Int) -> [UInt32] {
        guard end > start else { return [] }
        return (start..<end).map { i in
            buffer.contents().load(fromByteOffset: i * MemoryLayout<Float>.size, as: UInt32.self)
        }
    }

    /// Mixed absolute/relative comparison: |gpu − cpu| ≤ 1e-6 + 1e-5·|cpu|.
    private func expectClose(_ gpu: [Float], _ cpu: [Float], _ label: String) {
        #expect(gpu.count == cpu.count, "\(label): length \(gpu.count) vs \(cpu.count)")
        guard gpu.count == cpu.count else { return }
        var mismatches = 0
        var worst: Float = 0
        for i in 0..<gpu.count {
            let error = abs(gpu[i] - cpu[i])
            if !(error <= 1e-6 + 1e-5 * abs(cpu[i])) {
                mismatches += 1
            }
            worst = max(worst, error)
        }
        #expect(mismatches == 0, "\(label): \(mismatches) of \(gpu.count) differ, worst |Δ| \(worst)")
    }

    // MARK: - Tests

    @Test func matchesCPUAdamOverThreeSteps() throws {
        let context = try makeTestContext()
        let shDegree = 1
        let count = 1_000
        let cloud = try makeRandomCloud(count: count, seed: 5, shDegree: shDegree)
        let parameters = try GaussianBuffers(context: context, cloud: cloud)
        let gradients = try GradientBuffers(context: context, count: count, shDegree: shDegree)
        let state = try AdamBuffers(context: context, capacity: count, shDegree: shDegree)
        let adam = try GPUAdam(context: context)
        let rates = makeRates()
        let hyper = AdamHyperparameters()

        var cpuCloud = cloud
        var cpuState = AdamState(zeroLike: cloud)
        var rng = SplitMix64(seed: 5)
        for step in 1...3 {
            let grads = makeRandomGradients(count: count, shDegree: shDegree, rng: &rng)
            writeGradients(grads, to: gradients)

            let commandBuffer = try makeCommandBuffer(context)
            try adam.encode(parameters: parameters, gradients: gradients, state: state, rates: rates,
                            step: step, hyper: hyper, commandBuffer: commandBuffer)
            runAndWait(commandBuffer)

            cpuCloud = CPUAdam.step(cpuCloud, gradients: grads, state: &cpuState, rates: rates, hyper: hyper)

            let gpuParameters = parameters.snapshotArrays()
            expectClose(gpuParameters.positions, cpuCloud.positions, "step \(step) positions")
            expectClose(gpuParameters.logScales, cpuCloud.logScales, "step \(step) logScales")
            expectClose(gpuParameters.rotations, cpuCloud.rotations, "step \(step) rotations")
            expectClose(gpuParameters.opacityLogits, cpuCloud.opacityLogits, "step \(step) opacityLogits")
            expectClose(gpuParameters.sh, cpuCloud.sh, "step \(step) sh")

            let gpuM = state.m.snapshotArrays()
            let gpuV = state.v.snapshotArrays()
            expectClose(gpuM.positions, cpuState.m.positions, "step \(step) m.positions")
            expectClose(gpuM.sh, cpuState.m.sh, "step \(step) m.sh")
            expectClose(gpuV.positions, cpuState.v.positions, "step \(step) v.positions")
            expectClose(gpuV.sh, cpuState.v.sh, "step \(step) v.sh")
            expectClose(gpuV.rotations, cpuState.v.rotations, "step \(step) v.rotations")
        }
    }

    @Test func onlyFirstCountGaussiansChange() throws {
        let context = try makeTestContext()
        let shDegree = 1
        let count = 500
        let capacity = 2 * count
        let cloud = try makeRandomCloud(count: count, seed: 8, shDegree: shDegree)
        let parameters = try GaussianBuffers(context: context, capacity: capacity, shDegree: shDegree)
        parameters.upload(cloud)
        let state = try AdamBuffers(context: context, capacity: capacity, shDegree: shDegree)
        let gradients = try GradientBuffers(context: context, count: capacity, shDegree: shDegree)
        var rng = SplitMix64(seed: 9)
        writeGradients(makeRandomGradients(count: capacity, shDegree: shDegree, rng: &rng), to: gradients)

        // Sentinels in the spare capacity of parameters and moments.
        let floats = GaussianBuffers.floatsPerGaussian(shDegree: shDegree)
        for (group, buffer) in parameters.groups.enumerated() {
            fillFloats(buffer, from: count * floats[group], to: capacity * floats[group], value: 3.25)
        }
        for (group, buffer) in state.m.groups.enumerated() {
            fillFloats(buffer, from: count * floats[group], to: capacity * floats[group], value: 7.5)
        }
        for (group, buffer) in state.v.groups.enumerated() {
            fillFloats(buffer, from: count * floats[group], to: capacity * floats[group], value: 0.25)
        }

        let parameterTailBefore = parameters.groups.enumerated().map { group, buffer in
            floatBits(buffer, from: count * floats[group], to: capacity * floats[group])
        }
        let mTailBefore = state.m.groups.enumerated().map { group, buffer in
            floatBits(buffer, from: count * floats[group], to: capacity * floats[group])
        }
        let vTailBefore = state.v.groups.enumerated().map { group, buffer in
            floatBits(buffer, from: count * floats[group], to: capacity * floats[group])
        }

        let adam = try GPUAdam(context: context)
        let commandBuffer = try makeCommandBuffer(context)
        try adam.encode(parameters: parameters, gradients: gradients, state: state, rates: makeRates(),
                        step: 1, hyper: AdamHyperparameters(), commandBuffer: commandBuffer)
        runAndWait(commandBuffer)

        let parameterTailAfter = parameters.groups.enumerated().map { group, buffer in
            floatBits(buffer, from: count * floats[group], to: capacity * floats[group])
        }
        let mTailAfter = state.m.groups.enumerated().map { group, buffer in
            floatBits(buffer, from: count * floats[group], to: capacity * floats[group])
        }
        let vTailAfter = state.v.groups.enumerated().map { group, buffer in
            floatBits(buffer, from: count * floats[group], to: capacity * floats[group])
        }

        #expect(parameterTailAfter == parameterTailBefore, "parameters beyond count changed")
        #expect(mTailAfter == mTailBefore, "first moments beyond count changed")
        #expect(vTailAfter == vTailBefore, "second moments beyond count changed")
        #expect(parameters.count == count)
        // Sanity: the live part did change.
        #expect(parameters.snapshotArrays().positions != cloud.positions)
    }

    @Test func shHeadAndTailRates() throws {
        let context = try makeTestContext()
        let shDegree = 1
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        let cloud = try GaussianCloud(shDegree: shDegree,
                                      positions: [0, 0, 0],
                                      logScales: [0, 0, 0],
                                      rotations: [1, 0, 0, 0],
                                      opacityLogits: [0],
                                      sh: [Float](repeating: 0, count: 3 * k))
        let parameters = try GaussianBuffers(context: context, cloud: cloud)
        let gradients = try GradientBuffers(context: context, count: 1, shDegree: shDegree)
        // Positive gradient everywhere in SH, zero elsewhere: the first Adam step moves each SH float
        // by exactly −rate (sign of the gradient), with the DC rate for the first 3 floats.
        writeGradients(GaussianGradients(positions: [0, 0, 0], logScales: [0, 0, 0], rotations: [0, 0, 0, 0],
                                         opacityLogits: [0], sh: [Float](repeating: 1, count: 3 * k)),
                       to: gradients)
        let state = try AdamBuffers(context: context, capacity: 1, shDegree: shDegree)
        let adam = try GPUAdam(context: context)
        let rates = makeRates()

        let commandBuffer = try makeCommandBuffer(context)
        try adam.encode(parameters: parameters, gradients: gradients, state: state, rates: rates,
                        step: 1, hyper: AdamHyperparameters(), commandBuffer: commandBuffer)
        runAndWait(commandBuffer)

        let sh = parameters.snapshotArrays().sh
        #expect(sh.count == 3 * k)
        for i in 0..<sh.count {
            let expected: Float = i < 3 ? -rates.shDC : -rates.shRest
            let tolerance: Float = 1e-4 * abs(expected)
            #expect(abs(sh[i] - expected) <= tolerance, "sh[\(i)] = \(sh[i]), expected \(expected)")
        }
    }
}
