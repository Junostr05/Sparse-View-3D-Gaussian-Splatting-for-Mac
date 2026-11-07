import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU appearance compensation (v1.2) against the CPU reference `AppearanceTransform` and `AppearanceState`:
/// forward, backward (image gradient and parameter reduction) and the Adam update of one view.
@Suite(.serialized, .enabled(if: TestEnvironment.hasMetal))
struct GPUAppearanceTests {

    private func randomImage(width: Int, height: Int, seed: UInt64) -> RGBImage {
        var rng = SplitMix64(seed: seed)
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(width * height)
        for _ in 0..<(width * height) {
            let r = Float.random(in: 0.05...0.95, using: &rng)
            let g = Float.random(in: 0.05...0.95, using: &rng)
            let b = Float.random(in: 0.05...0.95, using: &rng)
            pixels.append(SIMD3<Float>(r, g, b))
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    /// An rgba32Float texture holding `image` (alpha 0.5, to show that the kernels do not depend on it).
    private func makeRenderTexture(context: MetalContext, image: RGBImage) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: image.width, height: image.height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        var texels = [Float](repeating: 0.5, count: image.pixels.count * 4)
        for i in 0..<image.pixels.count {
            texels[4 * i] = image.pixels[i].x
            texels[4 * i + 1] = image.pixels[i].y
            texels[4 * i + 2] = image.pixels[i].z
        }
        texels.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, image.width, image.height), mipmapLevel: 0,
                            withBytes: bytes.baseAddress!, bytesPerRow: image.width * 16)
        }
        return texture
    }

    private func readTexture(_ texture: MTLTexture) -> RGBImage {
        let count: Int = texture.width * texture.height
        var texels = [Float](repeating: 0, count: count * 4)
        texels.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: texture.width * 16,
                             from: MTLRegionMake2D(0, 0, texture.width, texture.height), mipmapLevel: 0)
        }
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(count)
        for i in 0..<count {
            let x: Float = texels[4 * i]
            let y: Float = texels[4 * i + 1]
            let z: Float = texels[4 * i + 2]
            pixels.append(SIMD3<Float>(x, y, z))
        }
        return RGBImage(width: texture.width, height: texture.height, pixels: pixels)
    }

    /// A buffer of RGBA floats per pixel holding `values` (alpha written as 0.25, which the kernel must overwrite).
    private func makeGradientBuffer(context: MetalContext, values: [SIMD3<Float>]) throws -> MTLBuffer {
        let buffer = try context.makeBuffer(length: values.count * 16)
        let pointer = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: values.count)
        for i in 0..<values.count {
            pointer[i] = SIMD4<Float>(values[i].x, values[i].y, values[i].z, 0.25)
        }
        return buffer
    }

    private func readGradientBuffer(_ buffer: MTLBuffer, count: Int) -> [SIMD4<Float>] {
        let pointer = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func parameters(_ gain: SIMD3<Float>, _ bias: SIMD3<Float>) -> AppearanceParameters {
        AppearanceParameters(gain: gain, bias: bias)
    }

    private func assertClose(_ a: SIMD3<Float>, _ b: SIMD3<Float>, tolerance: Float, _ label: String) {
        for c in 0..<3 {
            let scale: Float = max(1, abs(b[c]))
            #expect(abs(a[c] - b[c]) <= tolerance * scale, "\(label) channel \(c): gpu \(a[c]), cpu \(b[c])")
        }
    }

    @Test func forwardMatchesCPUReference() throws {
        let context = try MetalContext.makeDefault()
        let appearance = try GPUAppearance(context: context, viewCount: 3)
        let start = parameters(SIMD3<Float>(1.3, 0.8, 1.1), SIMD3<Float>(0.05, -0.04, 0.02))
        appearance.testingSetParameters([.identity, start, .identity])

        for (width, height) in [(13, 7), (160, 120)] {
            let image = randomImage(width: width, height: height, seed: UInt64(width * 31 + height))
            let texture = try makeRenderTexture(context: context, image: image)
            let queue = context.queue
            guard let commandBuffer = queue.makeCommandBuffer() else {
                Issue.record("no command buffer")
                return
            }
            let output = try appearance.encodeCompensation(rendered: texture, viewIndex: 1,
                                                           commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.status != .error)

            let gpu = readTexture(output)
            let cpu = AppearanceTransform.apply(image, start)
            var worst: Float = 0
            for i in 0..<cpu.pixels.count {
                let d = gpu.pixels[i] - cpu.pixels[i]
                worst = max(worst, abs(d.x), abs(d.y), abs(d.z))
            }
            #expect(worst <= 1e-5, "forward \(width)x\(height): worst difference \(worst)")
        }
    }

    @Test func backwardAndUpdateMatchCPUReference() throws {
        let context = try MetalContext.makeDefault()
        let width = 160
        let height = 120   // 19 200 pixels: more than one backward stride (64 · 256), so the loop strides twice
        let start = parameters(SIMD3<Float>(1.2, 0.85, 1.1), SIMD3<Float>(0.04, -0.03, 0.02))
        let appearance = try GPUAppearance(context: context, viewCount: 3)
        appearance.testingSetParameters([.identity, start, .identity])

        let rendered = randomImage(width: width, height: height, seed: 5)
        let texture = try makeRenderTexture(context: context, image: rendered)
        let count = width * height
        // Signed gradients in [-0.5, 0.5).
        let centred: SIMD3<Float> = SIMD3<Float>(repeating: 0.5)
        let imageOne: RGBImage = randomImage(width: width, height: height, seed: 6)
        let imageTwo: RGBImage = randomImage(width: width, height: height, seed: 8)
        var gradientOne: [SIMD3<Float>] = []
        var gradientTwo: [SIMD3<Float>] = []
        for i in 0..<count {
            let one: SIMD3<Float> = imageOne.pixels[i]
            let two: SIMD3<Float> = imageTwo.pixels[i]
            gradientOne.append(one - centred)
            gradientTwo.append(two - centred)
        }
        let bufferOne = try makeGradientBuffer(context: context, values: gradientOne)
        let bufferTwo = try makeGradientBuffer(context: context, values: gradientTwo)

        let learningRate: Float = 0.01
        let identityWeight: Float = 0.05
        let hyper = AdamHyperparameters()
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("no command buffer")
            return
        }
        try appearance.encodeGradient(rendered: texture, imageGradient: bufferOne, viewIndex: 1,
                                      learningRate: learningRate, identityWeight: identityWeight, hyper: hyper,
                                      commandBuffer: commandBuffer)
        try appearance.encodeGradient(rendered: texture, imageGradient: bufferTwo, viewIndex: 1,
                                      learningRate: learningRate, identityWeight: identityWeight, hyper: hyper,
                                      commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error)

        // CPU reference, step by step: each step's backward uses the parameters left by the previous update.
        var state = AppearanceState(parameters: start)
        let renderedImage = rendered
        let backwardOne = AppearanceTransform.backward(rendered: renderedImage, imageGradient: gradientOne,
                                                       parameters: state.parameters)
        state.apply(backwardOne.parameterGradient, identityWeight: identityWeight, learningRate: learningRate,
                    hyper: hyper)
        let backwardTwo = AppearanceTransform.backward(rendered: renderedImage, imageGradient: gradientTwo,
                                                       parameters: state.parameters)
        state.apply(backwardTwo.parameterGradient, identityWeight: identityWeight, learningRate: learningRate,
                    hyper: hyper)

        // Image gradients: dL/dR = G ⊙ gain of the step's parameters, alpha 0.
        let gpuOne = readGradientBuffer(bufferOne, count: count)
        let gpuTwo = readGradientBuffer(bufferTwo, count: count)
        var worstImage: Float = 0
        for i in 0..<count {
            let cpuOne = backwardOne.renderedGradient[i]
            let cpuTwo = backwardTwo.renderedGradient[i]
            worstImage = max(worstImage, abs(gpuOne[i].x - cpuOne.x), abs(gpuOne[i].y - cpuOne.y),
                             abs(gpuOne[i].z - cpuOne.z), abs(gpuTwo[i].x - cpuTwo.x),
                             abs(gpuTwo[i].y - cpuTwo.y), abs(gpuTwo[i].z - cpuTwo.z))
            #expect(gpuOne[i].w == 0 && gpuTwo[i].w == 0)
        }
        #expect(worstImage <= 1e-5, "image gradient: worst difference \(worstImage)")

        // Parameter reduction of the most recent backward (step two): Σ G ⊙ R and Σ G.
        let sums = appearance.partialSums()
        assertClose(SIMD3<Float>(sums[0], sums[1], sums[2]), backwardTwo.parameterGradient.gain, tolerance: 1e-4,
                    "gain gradient")
        assertClose(SIMD3<Float>(sums[3], sums[4], sums[5]), backwardTwo.parameterGradient.bias, tolerance: 1e-4,
                    "bias gradient")

        // Parameters after two Adam steps; the other views are untouched.
        let gpu = appearance.readParameters()
        assertClose(gpu[1].gain, state.parameters.gain, tolerance: 1e-4, "gain after two steps")
        assertClose(gpu[1].bias, state.parameters.bias, tolerance: 1e-4, "bias after two steps")
        #expect(gpu[0] == AppearanceParameters.identity)
        #expect(gpu[2] == AppearanceParameters.identity)
    }
}
