import Foundation
import Metal
import Testing
@testable import SplatCore
@testable import SplatMetal

/// v1.5 normal prior loss on the GPU against the CPU reference `NormalLoss` (itself checked against finite
/// differences in `NormalLossTests`).
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPUNormalLossTests {

    private struct Inputs {
        var draw: [Float] = []
        var alpha: [Float] = []
        var normals: [SIMD3<Float>] = []
    }

    /// A smooth depth surface with noise, opacity in [0.2, 1) (some pixels below the mask), normals tilted around
    /// (0, 0, −1) and a few pixels without a prior.
    private func makeInputs(width: Int, height: Int, seed: UInt64) -> Inputs {
        var rng = SplitMix64(seed: seed)
        var inputs = Inputs()
        for y in 0..<height {
            for x in 0..<width {
                let smooth: Float = 2 + 0.01 * Float(x) + 0.02 * Float(y) + 0.3 * sin(Float(x) * 0.2)
                let depthValue: Float = smooth + 0.02 * Float.random(in: -1..<1, using: &rng)
                let opacity: Float = Float.random(in: 0.2..<1, using: &rng)
                inputs.draw.append(opacity * depthValue)
                inputs.alpha.append(opacity)
                if (x + 3 * y) % 19 == 0 {
                    inputs.normals.append(.zero)
                } else {
                    let raw = SIMD3<Float>(Float.random(in: -0.4..<0.4, using: &rng),
                                           Float.random(in: -0.4..<0.4, using: &rng), -1)
                    inputs.normals.append(raw / (raw * raw).sum().squareRoot())
                }
            }
        }
        return inputs
    }

    private func run(context: MetalContext, loss: GPUNormalLoss, inputs: Inputs, width: Int, height: Int,
                     intrinsics: NormalLossIntrinsics, weights: NormalLossWeights,
                     initialGradient: [SIMD2<Float>]) throws -> (value: NormalLossValue, gradient: [SIMD2<Float>]) {
        let count: Int = width * height
        let render = try context.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride)
        let renderPointer = render.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        let gradientBuffer = try context.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride)
        let gradientPointer = gradientBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        for i in 0..<count {
            renderPointer[i] = SIMD2<Float>(inputs.draw[i], inputs.alpha[i])
            gradientPointer[i] = initialGradient[i]
        }
        let prior = try loss.makePrior(normals: inputs.normals, width: width, height: height)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try loss.encode(renderedDepth: render, prior: prior, intrinsics: intrinsics, weights: weights,
                        depthGradient: gradientBuffer, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error)
        var gradient: [SIMD2<Float>] = []
        gradient.reserveCapacity(count)
        for i in 0..<count {
            gradient.append(gradientPointer[i])
        }
        return (loss.readValue(), gradient)
    }

    @Test func matchesCPUAndAddsIntoTheDepthGradient() throws {
        let context = try makeTestContext()
        let loss = try GPUNormalLoss(context: context)
        let width: Int = 45
        let height: Int = 33
        let count: Int = width * height
        let inputs = makeInputs(width: width, height: height, seed: 909)
        let intrinsics = NormalLossIntrinsics(fx: 40, fy: 42, cx: 22.1, cy: 16.7)
        let weights = NormalLossWeights(weight: 0.05, alphaMask: 0.3, minimumPixels: 64)
        let cpu = NormalLoss.evaluate(depth: DepthRender(raw: inputs.draw, alpha: inputs.alpha), normals: inputs.normals,
                                      width: width, height: height, intrinsics: intrinsics, weights: weights)
        #expect(!cpu.degenerate)
        #expect(cpu.validPixels > 200)

        // Start from a non-zero gradient (as left by the depth loss) to check that the kernel adds.
        var rng = SplitMix64(seed: 5)
        var initial: [SIMD2<Float>] = []
        for _ in 0..<count {
            initial.append(SIMD2<Float>(Float.random(in: -1e-3..<1e-3, using: &rng),
                                        Float.random(in: -1e-3..<1e-3, using: &rng)))
        }
        let gpu = try run(context: context, loss: loss, inputs: inputs, width: width, height: height,
                          intrinsics: intrinsics, weights: weights, initialGradient: initial)
        #expect(gpu.value.validPixels == cpu.validPixels)
        #expect(gpu.value.degenerate == cpu.degenerate)
        #expect(abs(gpu.value.loss - cpu.loss) <= 1e-4 * abs(cpu.loss) + 1e-6, "loss gpu \(gpu.value.loss) cpu \(cpu.loss)")
        #expect(abs(gpu.value.meanCosine - cpu.meanCosine) <= 1e-4)

        var peak: Float = 0
        for g in cpu.gradient {
            peak = max(peak, max(abs(g.x), abs(g.y)))
        }
        #expect(peak > 0)
        var failures: Int = 0
        var first: String = ""
        for i in 0..<count {
            for c in 0..<2 {
                let expected: Float = initial[i][c] + cpu.gradient[i][c]
                let got: Float = gpu.gradient[i][c]
                let tolerance: Float = 1e-3 * abs(cpu.gradient[i][c]) + 1e-5 * peak + 1e-9
                if !(abs(got - expected) <= tolerance) {
                    failures += 1
                    if first.isEmpty {
                        first = "pixel \(i) component \(c): gpu \(got), expected \(expected)"
                    }
                }
            }
        }
        #expect(failures == 0, "\(failures) gradient components outside tolerance; first: \(first)")
    }

    @Test func degenerateLeavesTheGradientUntouched() throws {
        let context = try makeTestContext()
        let loss = try GPUNormalLoss(context: context)
        let width: Int = 8
        let height: Int = 8
        let inputs = makeInputs(width: width, height: height, seed: 3)
        let initial = [SIMD2<Float>](repeating: SIMD2<Float>(0.25, -0.5), count: width * height)
        let gpu = try run(context: context, loss: loss, inputs: inputs, width: width, height: height,
                          intrinsics: NormalLossIntrinsics(fx: 10, fy: 10, cx: 4, cy: 4),
                          weights: NormalLossWeights(weight: 1, alphaMask: 0.3, minimumPixels: 1_000),
                          initialGradient: initial)
        #expect(gpu.value.degenerate)
        #expect(gpu.value.loss == 0)
        #expect(gpu.gradient == initial)
    }
}
