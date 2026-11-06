import Foundation
import Metal
import Testing
@testable import SplatCore
@testable import SplatMetal

/// GPU Pearson + L1 depth loss (Phase 4 P7) against `DepthLoss.evaluate` (CPU reference), the
/// PyTorch Pearson golden, and slot independence / large-offset moment accuracy.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPUDepthLossTests {

    // MARK: - Helpers

    private static let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3,
                                                  minimumPixels: 64, nearPlane: 0.01)

    private struct RandomInputs {
        var draw: [Float]
        var alpha: [Float]
        var target: [Float]
        var metric: [Float]
    }

    /// Random view: depth in [1, 4), opacity in [0.2, 1) (some pixels fall below the 0.3 mask),
    /// a prior in the requested domain with a few invalid (≤ 0) targets, and a positive metric.
    private func makeRandomInputs(count: Int, seed: UInt64, inverseTarget: Bool) -> RandomInputs {
        var rng = SplitMix64(seed: seed)
        var inputs = RandomInputs(draw: [], alpha: [], target: [], metric: [])
        for i in 0..<count {
            let depthValue: Float = Float.random(in: 1..<4, using: &rng)
            let opacity: Float = Float.random(in: 0.2..<1, using: &rng)
            let noise: Float = Float.random(in: -1..<1, using: &rng)
            var targetValue: Float
            if inverseTarget {
                targetValue = 1 / depthValue + 0.05 * noise
            } else {
                targetValue = 2 * depthValue + 0.5 + 0.3 * noise
            }
            if i % 17 == 0 {
                targetValue = -1
            }
            inputs.draw.append(opacity * depthValue)
            inputs.alpha.append(opacity)
            inputs.target.append(targetValue)
            inputs.metric.append(1.1 * depthValue + 0.05)
        }
        return inputs
    }

    private func makePrior(width: Int, height: Int, inputs: RandomInputs, domain: DepthDomain,
                           measured: Bool) -> DepthPrior {
        DepthPrior(width: width, height: height, correlationTarget: inputs.target, domain: domain,
                   metricDepth: inputs.metric, isMeasuredMetric: measured)
    }

    /// Interleaved `(Draw, A)` per pixel, as Task 14 writes the rendered depth.
    private func makeRenderBuffer(_ context: MetalContext, draw: [Float], alpha: [Float]) throws -> MTLBuffer {
        let count: Int = draw.count
        let buffer = try context.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride)
        let pointer = buffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        for i in 0..<count {
            pointer[i] = SIMD2<Float>(draw[i], alpha[i])
        }
        return buffer
    }

    private func makeGradientBuffer(_ context: MetalContext, count: Int) throws -> MTLBuffer {
        try context.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride)
    }

    private func readGradient(_ buffer: MTLBuffer, count: Int) -> [SIMD2<Float>] {
        let pointer = buffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(count)
        for i in 0..<count {
            out.append(pointer[i])
        }
        return out
    }

    /// Encodes one loss into `slot` in its own command buffer and reads value and gradient back.
    private func runGPU(context: MetalContext, loss: GPUDepthLoss, prior: GPUDepthPrior,
                        render: MTLBuffer, count: Int, weights: DepthLossWeights,
                        slot: Int) throws -> (value: DepthLossValue, gradient: [SIMD2<Float>]) {
        let gradientBuffer: MTLBuffer = try makeGradientBuffer(context, count: count)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try loss.encode(renderedDepth: render, prior: prior, weights: weights, slot: slot,
                        depthGradient: gradientBuffer, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error, "depth loss command buffer failed")
        return (loss.readValue(slot: slot), readGradient(gradientBuffer, count: count))
    }

    private func cpuReference(draw: [Float], alpha: [Float], prior: DepthPrior,
                              weights: DepthLossWeights) -> DepthLossResult {
        DepthLoss.evaluate(depth: DepthRender(raw: draw, alpha: alpha), prior: prior, weights: weights)
    }

    /// Scalar comparison: 1e-4 relative with a 1e-6 absolute floor.
    private func expectScalar(_ gpu: Double, _ cpu: Double, _ name: String) {
        let tolerance: Double = 1e-4 * abs(cpu) + 1e-6
        #expect(abs(gpu - cpu) <= tolerance, "\(name): gpu \(gpu), cpu \(cpu)")
    }

    /// Gradient comparison: |gpu − cpu| ≤ 1e-7 + 1e-3·|cpu| per component.
    private func expectGradient(_ gpu: [SIMD2<Float>], _ cpu: [SIMD2<Float>], _ name: String) {
        #expect(gpu.count == cpu.count, "\(name): gradient count \(gpu.count) vs \(cpu.count)")
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

    private func expectValueMatchesCPU(_ value: DepthLossValue, _ cpu: DepthLossResult, _ name: String) {
        expectScalar(value.loss, cpu.loss, "\(name) loss")
        expectScalar(value.pearson, cpu.pearson, "\(name) pearson")
        expectScalar(value.l1, cpu.l1, "\(name) l1")
        #expect(value.validPixels == cpu.validPixels, "\(name): validPixels \(value.validPixels) vs \(cpu.validPixels)")
        #expect(value.degenerate == cpu.degenerate, "\(name): degenerate flag differs")
    }

    // MARK: - Tests

    @Test func matchesCPUOnRandomInputs() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let width: Int = 37
        let height: Int = 23
        let count: Int = width * height

        for domain in [DepthDomain.depth, DepthDomain.inverseDepth] {
            let inputs = makeRandomInputs(count: count, seed: 101, inverseTarget: domain == .inverseDepth)
            let prior = makePrior(width: width, height: height, inputs: inputs, domain: domain, measured: true)
            let gpuPrior = try loss.makePrior(prior)
            let render = try makeRenderBuffer(context, draw: inputs.draw, alpha: inputs.alpha)

            let cpu = cpuReference(draw: inputs.draw, alpha: inputs.alpha, prior: prior, weights: Self.weights)
            #expect(!cpu.degenerate)
            #expect(cpu.l1 > 0)

            let gpu = try runGPU(context: context, loss: loss, prior: gpuPrior, render: render, count: count,
                                 weights: Self.weights, slot: 0)
            let name = "\(domain)"
            expectValueMatchesCPU(gpu.value, cpu, name)
            expectGradient(gpu.gradient, cpu.gradient, name)
        }
    }

    @Test func matchesPearsonGolden() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let golden = try PearsonGolden.load("pearson")
        let weights = DepthLossWeights(pearson: golden.lambdaPearson, l1: golden.lambdaL1,
                                       alphaMask: golden.alphaMask, minimumPixels: golden.minimumPixels,
                                       nearPlane: golden.nearPlane)
        let count: Int = golden.width * golden.height
        #expect(golden.draw.count == count)
        #expect(golden.cases.count == 2)
        let render = try makeRenderBuffer(context, draw: golden.draw, alpha: golden.alpha)

        for c in golden.cases {
            let domain: DepthDomain = c.domain == "depth" ? .depth : .inverseDepth
            let prior = DepthPrior(width: golden.width, height: golden.height, correlationTarget: c.target,
                                   domain: domain, metricDepth: golden.metricDepth,
                                   isMeasuredMetric: golden.isMeasuredMetric)
            let gpuPrior = try loss.makePrior(prior)
            let gpu = try runGPU(context: context, loss: loss, prior: gpuPrior, render: render, count: count,
                                 weights: weights, slot: 0)

            #expect(gpu.value.validPixels == c.validPixels, "\(c.domain): validPixels")
            #expect(gpu.value.degenerate == c.degenerate, "\(c.domain): degenerate flag")
            expectScalar(gpu.value.loss, c.loss, "\(c.domain) golden loss")
            expectScalar(gpu.value.pearson, c.pearson, "\(c.domain) golden pearson")
            expectScalar(gpu.value.l1, c.l1, "\(c.domain) golden l1")

            var peak: Float = 0
            for g in c.gradDraw { peak = max(peak, abs(g)) }
            for g in c.gradA { peak = max(peak, abs(g)) }
            let floorValue: Float = 1e-6 * peak
            #expect(gpu.gradient.count == count)
            var failures: Int = 0
            for p in 0..<count {
                let refDraw: Float = c.gradDraw[p]
                let refA: Float = c.gradA[p]
                if !(abs(gpu.gradient[p].x - refDraw) <= 1e-3 * abs(refDraw) + floorValue) { failures += 1 }
                if !(abs(gpu.gradient[p].y - refA) <= 1e-3 * abs(refA) + floorValue) { failures += 1 }
            }
            #expect(failures == 0, "\(c.domain) golden gradient: \(failures) components outside tolerance")
        }
    }

    @Test func gradientMatchesFiniteDifferences() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let width: Int = 16
        let height: Int = 12
        let count: Int = width * height
        let step: Double = 1e-4

        for domain in [DepthDomain.depth, DepthDomain.inverseDepth] {
            let inputs = makeRandomInputs(count: count, seed: 401, inverseTarget: domain == .inverseDepth)
            let prior = makePrior(width: width, height: height, inputs: inputs, domain: domain, measured: true)
            let gpuPrior = try loss.makePrior(prior)
            let render = try makeRenderBuffer(context, draw: inputs.draw, alpha: inputs.alpha)
            let gpu = try runGPU(context: context, loss: loss, prior: gpuPrior, render: render, count: count,
                                 weights: Self.weights, slot: 0)

            // Central differences in Double on the same Float-rounded inputs. The mask is fixed under the
            // perturbations because opacity and target are not perturbed, only Draw or A.
            let drawD: [Double] = inputs.draw.map { Double($0) }
            let alphaD: [Double] = inputs.alpha.map { Double($0) }
            var failures: Int = 0
            for p in 0..<count {
                var drawPlus: [Double] = drawD
                var drawMinus: [Double] = drawD
                drawPlus[p] += step
                drawMinus[p] -= step
                let lossDrawPlus: Double = DepthLoss.evaluateDouble(draw: drawPlus, alpha: alphaD, prior: prior,
                                                                    weights: Self.weights).loss
                let lossDrawMinus: Double = DepthLoss.evaluateDouble(draw: drawMinus, alpha: alphaD, prior: prior,
                                                                     weights: Self.weights).loss
                let fdDraw: Double = (lossDrawPlus - lossDrawMinus) / (2 * step)

                var alphaPlus: [Double] = alphaD
                var alphaMinus: [Double] = alphaD
                alphaPlus[p] += step
                alphaMinus[p] -= step
                let lossAlphaPlus: Double = DepthLoss.evaluateDouble(draw: drawD, alpha: alphaPlus, prior: prior,
                                                                     weights: Self.weights).loss
                let lossAlphaMinus: Double = DepthLoss.evaluateDouble(draw: drawD, alpha: alphaMinus, prior: prior,
                                                                      weights: Self.weights).loss
                let fdAlpha: Double = (lossAlphaPlus - lossAlphaMinus) / (2 * step)

                let gotDraw: Double = Double(gpu.gradient[p].x)
                let gotAlpha: Double = Double(gpu.gradient[p].y)
                if !(abs(gotDraw - fdDraw) <= 1e-3 * abs(fdDraw) + 1e-8) { failures += 1 }
                if !(abs(gotAlpha - fdAlpha) <= 1e-3 * abs(fdAlpha) + 1e-8) { failures += 1 }
            }
            #expect(failures == 0, "\(domain): \(failures) gradient components disagree with finite differences")
        }
    }

    @Test func gpuDegenerateCasesAreZero() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)

        // Constant rendered depth (zero variance) with a varying prior.
        let side: Int = 10
        let count: Int = side * side
        var constantTarget: [Float] = []
        var constantDraw: [Float] = []
        var constantAlpha: [Float] = []
        for i in 0..<count {
            constantTarget.append(Float(1 + 0.01 * Double(i)))
            constantDraw.append(2)
            constantAlpha.append(1)
        }
        let constantPrior = DepthPrior(width: side, height: side, correlationTarget: constantTarget, domain: .depth)

        // Empty mask: opacity below the threshold everywhere.
        var emptyDraw: [Float] = []
        var emptyAlpha: [Float] = []
        for _ in 0..<count {
            emptyDraw.append(0.1)
            emptyAlpha.append(0.1)
        }

        // Single pixel: fewer than minimumPixels.
        let singlePrior = DepthPrior(width: 1, height: 1, correlationTarget: [2], domain: .depth)

        let cases: [(name: String, prior: DepthPrior, draw: [Float], alpha: [Float])] = [
            ("constant", constantPrior, constantDraw, constantAlpha),
            ("empty", constantPrior, emptyDraw, emptyAlpha),
            ("single", singlePrior, [1.5], [1]),
        ]
        for c in cases {
            let pixelCount: Int = c.draw.count
            let gpuPrior = try loss.makePrior(c.prior)
            let render = try makeRenderBuffer(context, draw: c.draw, alpha: c.alpha)
            let gpu = try runGPU(context: context, loss: loss, prior: gpuPrior, render: render,
                                 count: pixelCount, weights: Self.weights, slot: 0)
            #expect(gpu.value.degenerate, "\(c.name): not flagged degenerate")
            #expect(gpu.value.loss.isFinite, "\(c.name): loss is not finite")
            #expect(gpu.value.loss == 0, "\(c.name): loss \(gpu.value.loss) is not zero")
            #expect(gpu.value.pearson == 0, "\(c.name): pearson \(gpu.value.pearson) is not zero")
            #expect(gpu.gradient.count == pixelCount)
            for g in gpu.gradient {
                #expect(!g.x.isNaN && !g.y.isNaN, "\(c.name): NaN gradient")
                #expect(g == .zero, "\(c.name): gradient is not zero")
            }
        }
        #expect(loss.readValue(slot: 0).validPixels == 1)
    }

    @Test func slotsAreIndependent() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let width: Int = 37
        let height: Int = 23
        let count: Int = width * height

        let inputsA = makeRandomInputs(count: count, seed: 201, inverseTarget: false)
        let inputsB = makeRandomInputs(count: count, seed: 202, inverseTarget: true)
        let priorA = makePrior(width: width, height: height, inputs: inputsA, domain: .depth, measured: false)
        let priorB = makePrior(width: width, height: height, inputs: inputsB, domain: .inverseDepth, measured: true)
        let gpuPriorA = try loss.makePrior(priorA)
        let gpuPriorB = try loss.makePrior(priorB)
        let renderA = try makeRenderBuffer(context, draw: inputsA.draw, alpha: inputsA.alpha)
        let renderB = try makeRenderBuffer(context, draw: inputsB.draw, alpha: inputsB.alpha)
        let gradientA = try makeGradientBuffer(context, count: count)
        let gradientB = try makeGradientBuffer(context, count: count)

        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try loss.encode(renderedDepth: renderA, prior: gpuPriorA, weights: Self.weights, slot: 0,
                        depthGradient: gradientA, commandBuffer: commandBuffer)
        try loss.encode(renderedDepth: renderB, prior: gpuPriorB, weights: Self.weights, slot: 1,
                        depthGradient: gradientB, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error, "depth loss command buffer failed")

        let cpuA = cpuReference(draw: inputsA.draw, alpha: inputsA.alpha, prior: priorA, weights: Self.weights)
        let cpuB = cpuReference(draw: inputsB.draw, alpha: inputsB.alpha, prior: priorB, weights: Self.weights)
        expectValueMatchesCPU(loss.readValue(slot: 0), cpuA, "slot 0")
        expectValueMatchesCPU(loss.readValue(slot: 1), cpuB, "slot 1")
        expectGradient(readGradient(gradientA, count: count), cpuA.gradient, "slot 0")
        expectGradient(readGradient(gradientB, count: count), cpuB.gradient, "slot 1")
    }

    @Test func largeImageMomentsAreAccurate() throws {
        let context = try makeTestContext()
        let loss = try GPUDepthLoss(context: context)
        let side: Int = 64
        let count: Int = side * side

        // Depth offset 1000 with a spread of about 0.5: the one-pass moment formula would cancel.
        var rng = SplitMix64(seed: 301)
        var draw: [Float] = []
        var alpha: [Float] = []
        var target: [Float] = []
        for _ in 0..<count {
            let spread: Float = Float.random(in: 0..<0.5, using: &rng)
            let noise: Float = Float.random(in: -0.1..<0.1, using: &rng)
            let depthValue: Float = 1000 + spread
            draw.append(depthValue)
            alpha.append(1)
            target.append(3 * spread + 1 + noise)
        }
        let prior = DepthPrior(width: side, height: side, correlationTarget: target, domain: .depth)
        let gpuPrior = try loss.makePrior(prior)
        let render = try makeRenderBuffer(context, draw: draw, alpha: alpha)
        let cpu = cpuReference(draw: draw, alpha: alpha, prior: prior, weights: Self.weights)
        #expect(!cpu.degenerate)

        let gpu = try runGPU(context: context, loss: loss, prior: gpuPrior, render: render, count: count,
                             weights: Self.weights, slot: 0)
        #expect(!gpu.value.degenerate)
        #expect(abs(gpu.value.pearson - cpu.pearson) <= 1e-4, "pearson gpu \(gpu.value.pearson), cpu \(cpu.pearson)")
        #expect(abs(gpu.value.loss - cpu.loss) <= 1e-4 * abs(cpu.loss) + 1e-6)
    }
}
