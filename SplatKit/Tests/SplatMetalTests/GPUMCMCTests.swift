import Metal
import Testing
import SplatCore
@testable import SplatMetal

// GPU MCMC kernels (v1.2) against the CPU references in SplatCore: the opacity and scale regularizer gradients
// and the SGLD position noise. The noise uses the counter-based generator on both sides, so no ε is injected
// on the GPU. Float atomics do not appear in these kernels, but the transcendental functions differ in the last
// bits, so the comparisons use absolute tolerances.

@Suite(.serialized, .enabled(if: TestEnvironment.hasMetal))
struct GPUMCMCTests {

    // MARK: - Helpers

    /// Random degree-0 cloud with varied scales, rotations and opacities.
    private func makeCloud(count: Int, seed: UInt64) throws -> GaussianCloud {
        var rng = SeededGenerator(seed: seed)
        var positions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var logits: [Float] = []
        var sh: [Float] = []
        for _ in 0..<count {
            for _ in 0..<3 {
                positions.append(Float.random(in: -1...1, using: &rng))
            }
            positions[positions.count - 1] += 2
            for _ in 0..<3 {
                logScales.append(Float.random(in: -4 ... -0.5, using: &rng))
            }
            for _ in 0..<4 {
                rotations.append(Float.random(in: -1...1, using: &rng))
            }
            logits.append(Float.random(in: -4...4, using: &rng))
            for _ in 0..<3 {
                sh.append(Float.random(in: -0.5...0.5, using: &rng))
            }
        }
        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: logits, sh: sh)
    }

    private func zero(_ gradients: GradientBuffers) {
        for buffer in [gradients.positions, gradients.logScales, gradients.rotations,
                       gradients.opacityLogits, gradients.sh] {
            memset(buffer.contents(), 0, buffer.length)
        }
    }

    /// Commits the buffer, waits for it and checks that it did not fail.
    private func run(_ context: MetalContext, _ encode: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try encode(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error)
    }

    // MARK: - Regularizers

    @Test func regularizerKernelMatchesCPU() throws {
        let context = try MetalContext.makeDefault()
        let cloud = try makeCloud(count: 301, seed: 5)
        let parameters = try GaussianBuffers(context: context, cloud: cloud)
        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: 0)
        zero(gradients)
        let gpu = try GPUMCMC(context: context)

        try run(context) { commandBuffer in
            try gpu.encodeRegularizers(parameters: parameters, gradients: gradients, opacityWeight: 0.3,
                                       scaleWeight: 0.2, commandBuffer: commandBuffer)
        }

        var reference = GaussianGradients(zeroLike: cloud)
        MCMCRegularizers.addGradients(opacityLogits: cloud.opacityLogits, logScales: cloud.logScales,
                                      opacityWeight: 0.3, scaleWeight: 0.2, into: &reference)
        let actual = gradients.toGaussianGradients(count: cloud.count, shDegree: 0)
        for (name, cpuValues, gpuValues) in [("opacity", reference.opacityLogits, actual.opacityLogits),
                                             ("logScale", reference.logScales, actual.logScales)] {
            #expect(cpuValues.count == gpuValues.count)
            var worst: Float = 0
            for i in 0..<min(cpuValues.count, gpuValues.count) {
                worst = max(worst, abs(cpuValues[i] - gpuValues[i]) / max(1e-3, abs(cpuValues[i])))
            }
            #expect(worst < 1e-4, "\(name) relative error \(worst)")
        }
    }

    // MARK: - Noise

    @Test func noiseKernelMatchesCPUReference() throws {
        let context = try MetalContext.makeDefault()
        let cloud = try makeCloud(count: 300, seed: 9)
        let parameters = try GaussianBuffers(context: context, cloud: cloud)
        let seed: UInt32 = 4_242
        let step: Float = 2e-2
        let gpu = try GPUMCMC(context: context)

        try run(context) { commandBuffer in
            try gpu.encodeNoise(parameters: parameters, step: step, seed: seed, commandBuffer: commandBuffer)
        }

        var reference = cloud.positions
        MCMCNoise.apply(positions: &reference, logScales: cloud.logScales, rotations: cloud.rotations,
                        opacityLogits: cloud.opacityLogits, step: step) { index in
            SIMD3<Float>(SplatRandom.normal(seed: seed, index: UInt32(index), stream: 0),
                         SplatRandom.normal(seed: seed, index: UInt32(index), stream: 1),
                         SplatRandom.normal(seed: seed, index: UInt32(index), stream: 2))
        }
        let actual = parameters.snapshot().positions
        #expect(actual.count == reference.count)
        var worst: Float = 0
        var largest: Float = 0
        for i in 0..<min(actual.count, reference.count) {
            worst = max(worst, abs(actual[i] - reference[i]))
            largest = max(largest, abs(reference[i] - cloud.positions[i]))
        }
        #expect(largest > 0, "the noise should move the positions")
        #expect(worst < 1e-5, "max absolute difference \(worst)")
    }
}
