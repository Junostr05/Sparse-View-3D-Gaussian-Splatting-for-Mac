import Foundation
import Metal
import Testing
@testable import SplatCore
@testable import SplatMetal

/// v1.5 confidence-weighted depth loss on the GPU against the CPU reference (`DepthLoss.evaluate`), which
/// is itself checked against finite differences in `WeightedDepthLossTests`.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPUWeightedDepthLossTests {

    private static let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3,
                                                  minimumPixels: 64, nearPlane: 0.01)

    private struct Inputs {
        var draw: [Float] = []
        var alpha: [Float] = []
        var target: [Float] = []
        var metric: [Float] = []
        var pixelWeights: [Float] = []
    }

    /// Depth in [1, 4), opacity in [0.2, 1) (some pixels fall below the mask), a few invalid targets, a
    /// positive metric and weights spread over two orders of magnitude, like Depth Anything 3 confidence.
    private func makeInputs(count: Int, seed: UInt64, inverseTarget: Bool) -> Inputs {
        var rng = SplitMix64(seed: seed)
        var inputs = Inputs()
        for i in 0..<count {
            let depthValue: Float = Float.random(in: 1..<4, using: &rng)
            let opacity: Float = Float.random(in: 0.2..<1, using: &rng)
            let noise: Float = Float.random(in: -1..<1, using: &rng)
            var targetValue: Float = inverseTarget ? 1 / depthValue + 0.05 * noise : 2 * depthValue + 0.5 + 0.3 * noise
            if i % 17 == 0 {
                targetValue = -1
            }
            inputs.draw.append(opacity * depthValue)
            inputs.alpha.append(opacity)
            inputs.target.append(targetValue)
            inputs.metric.append(1.1 * depthValue + 0.05 + 0.1 * noise)
            let exponent: Float = Float.random(in: 0..<2, using: &rng)
            inputs.pixelWeights.append(i % 23 == 0 ? 0 : Float(pow(10.0, Double(exponent))))
        }
        return inputs
    }

    private func makeRenderBuffer(_ context: MetalContext, draw: [Float], alpha: [Float]) throws -> MTLBuffer {
        let count: Int = draw.count
        let buffer = try context.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride)
        let pointer = buffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        for i in 0..<count {
            pointer[i] = SIMD2<Float>(draw[i], alpha[i])
        }
        return buffer
    }

    private func runGPU(context: MetalContext, loss: GPUDepthLoss, prior: DepthPrior, render: MTLBuffer,
                        count: Int, slot: Int) throws -> (value: DepthLossValue, gradient: [SIMD2<Float>]) {
        let gpuPrior = try loss.makePrior(prior)
        let gradientBuffer = try context.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try loss.encode(renderedDepth: render, prior: gpuPrior, weights: Self.weights, slot: slot,
                        depthGradient: gradientBuffer, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error, "depth loss command buffer failed")
        let pointer = gradientBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        var gradient: [SIMD2<Float>] = []
        gradient.reserveCapacity(count)
        for i in 0..<count {
            gradient.append(pointer[i])
        }
        return (loss.readValue(slot: slot), gradient)
    }

    private func expectScalar(_ gpu: Double, _ cpu: Double, _ name: String) {
        let tolerance: Double = 1e-4 * abs(cpu) + 1e-6
        #expect(abs(gpu - cpu) <= tolerance, "\(name): gpu \(gpu), cpu \(cpu)")
    }

    private func expectGradient(_ gpu: [SIMD2<Float>], _ cpu: [SIMD2<Float>], _ name: String) {
        #expect(gpu.count == cpu.count)
        guard gpu.count == cpu.count else { return }
        var failures: Int = 0
        var first: String = ""
        for i in gpu.indices {
            for c in 0..<2 {
                let reference: Float = cpu[i][c]
                let tolerance: Float = 1e-7 + 1e-3 * abs(reference)
                if !(abs(gpu[i][c] - reference) <= tolerance) {
                    failures += 1
                    if first.isEmpty {
                        first = "pixel \(i) component \(c): gpu \(gpu[i][c]), cpu \(reference)"
                    }
                }
            }
        }
        #expect(failures == 0, "\(name): \(failures) gradient components outside tolerance; first: \(first)")
    }

    @Test func weightedLossMatchesCPU() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let width: Int = 53
        let height: Int = 29
        let count: Int = width * height
        for domain in [DepthDomain.depth, DepthDomain.inverseDepth] {
            for measured in [false, true] {
                let inputs = makeInputs(count: count, seed: 211, inverseTarget: domain == .inverseDepth)
                let prior = DepthPrior(width: width, height: height, correlationTarget: inputs.target, domain: domain,
                                       metricDepth: inputs.metric, isMeasuredMetric: measured,
                                       weights: inputs.pixelWeights)
                #expect(prior.weights != nil)
                let render = try makeRenderBuffer(context, draw: inputs.draw, alpha: inputs.alpha)
                let cpu = DepthLoss.evaluate(depth: DepthRender(raw: inputs.draw, alpha: inputs.alpha), prior: prior,
                                             weights: Self.weights)
                #expect(!cpu.degenerate)
                let gpu = try runGPU(context: context, loss: loss, prior: prior, render: render, count: count, slot: 1)
                let name: String = "\(domain) measured \(measured)"
                expectScalar(gpu.value.loss, cpu.loss, "\(name) loss")
                expectScalar(gpu.value.pearson, cpu.pearson, "\(name) pearson")
                expectScalar(gpu.value.l1, cpu.l1, "\(name) l1")
                #expect(gpu.value.validPixels == cpu.validPixels)
                #expect(gpu.value.degenerate == cpu.degenerate)
                expectGradient(gpu.gradient, cpu.gradient, name)
            }
        }
    }

    /// The weights change the GPU result (the test above is not passing by ignoring them), and uniform
    /// weights reproduce the unweighted GPU result.
    @Test func weightsChangeTheResultAndUniformWeightsDoNot() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let width: Int = 40
        let height: Int = 30
        let count: Int = width * height
        let inputs = makeInputs(count: count, seed: 307, inverseTarget: false)
        let render = try makeRenderBuffer(context, draw: inputs.draw, alpha: inputs.alpha)
        let plain = DepthPrior(width: width, height: height, correlationTarget: inputs.target, domain: .depth,
                               metricDepth: inputs.metric, isMeasuredMetric: true)
        let uniform = DepthPrior(width: width, height: height, correlationTarget: inputs.target, domain: .depth,
                                 metricDepth: inputs.metric, isMeasuredMetric: true,
                                 weights: [Float](repeating: 7, count: count))
        let weighted = DepthPrior(width: width, height: height, correlationTarget: inputs.target, domain: .depth,
                                  metricDepth: inputs.metric, isMeasuredMetric: true, weights: inputs.pixelWeights)
        let a = try runGPU(context: context, loss: loss, prior: plain, render: render, count: count, slot: 0)
        let b = try runGPU(context: context, loss: loss, prior: uniform, render: render, count: count, slot: 0)
        let c = try runGPU(context: context, loss: loss, prior: weighted, render: render, count: count, slot: 0)
        expectScalar(b.value.loss, a.value.loss, "uniform loss")
        expectScalar(b.value.pearson, a.value.pearson, "uniform pearson")
        expectScalar(b.value.l1, a.value.l1, "uniform l1")
        expectGradient(b.gradient, a.gradient, "uniform")
        #expect(abs(c.value.loss - a.value.loss) > 1e-4 * abs(a.value.loss))
    }
}
