import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU camera-pose gradient (pose scratch in `projectBackward`, reduction in `reducePoseGradient`) against the CPU
/// reference `CPURasterizer.poseBackward`.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct PoseGradientMetalTests {

    // MARK: - Helpers

    private func axisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
        let half = angle / 2
        return SIMD4(cos(half), sin(half) * axis.x, sin(half) * axis.y, sin(half) * axis.z)
    }

    /// Pinhole camera with R = Rx(0.35)·Ry(−0.25) and t = (0.3, −0.2, 0.4).
    private func rotatedCamera(width: Int, height: Int) -> Camera {
        let rx = Matrix3.rotation(quaternion: axisAngle(SIMD3(1, 0, 0), 0.35))
        let ry = Matrix3.rotation(quaternion: axisAngle(SIMD3(0, 1, 0), -0.25))
        return Camera(width: width, height: height,
                      fx: Float(height), fy: Float(height),
                      cx: Float(width) / 2, cy: Float(height) / 2,
                      rotation: rx * ry, translation: SIMD3(0.3, -0.2, 0.4))
    }

    private func randomImageGradient(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in
            SIMD3(Float.random(in: -1...1, using: &rng),
                  Float.random(in: -1...1, using: &rng),
                  Float.random(in: -1...1, using: &rng))
        }
    }

    /// Image gradient as the float4-per-pixel buffer `encodeBackward` reads.
    private func gradientBuffer(_ values: [SIMD3<Float>], context: MetalContext) throws -> MTLBuffer {
        let buffer = try context.makeBuffer(length: max(1, values.count) * 4 * MemoryLayout<Float>.size)
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count * 4)
        for (i, v) in values.enumerated() {
            pointer[4 * i] = v.x
            pointer[4 * i + 1] = v.y
            pointer[4 * i + 2] = v.z
            pointer[4 * i + 3] = 0
        }
        return buffer
    }

    /// Renders `camera`, then encodes the backward pass (with a pose sink when `withSink`) and returns the six GPU
    /// sums (dL/dω, then dL/dτ) and the parameter gradients.
    private func gpuBackward(context: MetalContext, cloud: GaussianCloud, camera: Camera, settings: RasterSettings,
                             imageGradient: [SIMD3<Float>], withSink: Bool)
        throws -> (sums: [Float], gradients: GaussianGradients) {
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        _ = try renderer.renderImage(camera: camera, settings: settings)
        let gradientImage = try gradientBuffer(imageGradient, context: context)
        let count = cloud.count
        let gradients = try GradientBuffers(context: context, count: count, shDegree: cloud.shDegree)
        let reducer = try GPUPoseGradient(context: context)
        let scratch = try context.makeBuffer(length: max(1, count) * 6 * MemoryLayout<Float>.size)
        let sums = try context.makeBuffer(length: 6 * MemoryLayout<Float>.size)
        memset(sums.contents(), 0, sums.length)
        let sink = withSink ? PoseGradientSink(reducer: reducer, scratch: scratch, sums: sums, base: 0) : nil

        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try renderer.encodeBackward(imageGradient: gradientImage, depthGradient: nil, into: gradients,
                                    poseSink: sink, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error)

        let pointer = sums.contents().bindMemory(to: Float.self, capacity: 6)
        let values = Array(UnsafeBufferPointer(start: pointer, count: 6))
        return (values, gradients.toGaussianGradients(count: count, shDegree: cloud.shDegree))
    }

    /// Every entry within 1e-4 + 1e-3·|cpu| (as in BackwardTests).
    private func expectClose(_ gpu: [Float], _ cpu: [Float], _ name: String) {
        #expect(gpu.count == cpu.count, "\(name): count \(gpu.count) vs \(cpu.count)")
        guard gpu.count == cpu.count else { return }
        for i in gpu.indices {
            let tolerance: Float = 1e-4 + 1e-3 * abs(cpu[i])
            #expect(abs(gpu[i] - cpu[i]) <= tolerance,
                    "\(name)[\(i)]: gpu \(gpu[i]), cpu \(cpu[i])")
        }
    }

    private func expectMatchesCPU(context: MetalContext, cloud: GaussianCloud, camera: Camera,
                                  settings: RasterSettings, seed: UInt64) throws {
        let imageGradient = randomImageGradient(count: camera.width * camera.height, seed: seed)
        let gpu = try gpuBackward(context: context, cloud: cloud, camera: camera, settings: settings,
                                  imageGradient: imageGradient, withSink: true)
        let cpu = CPURasterizer.poseBackward(cloud, camera: camera, settings: settings,
                                             imageGradient: imageGradient)
        let cpuSums: [Float] = [Float(cpu.rotation.x), Float(cpu.rotation.y), Float(cpu.rotation.z),
                                Float(cpu.translation.x), Float(cpu.translation.y), Float(cpu.translation.z)]
        expectClose(gpu.sums, cpuSums, "pose")
    }

    // MARK: - Tests

    @Test func poseGradientMatchesCPUOnRotatedCamera() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 200, seed: 21, shDegree: 1)
        try expectMatchesCPU(context: context, cloud: cloud, camera: rotatedCamera(width: 64, height: 64),
                             settings: RasterSettings(), seed: 31)
    }

    @Test func poseGradientMatchesCPUOnIdentityCamera() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 150, seed: 5, shDegree: 1)
        try expectMatchesCPU(context: context, cloud: cloud, camera: makeTestCamera(width: 64, height: 64),
                             settings: RasterSettings(), seed: 32)
    }

    /// Largest difference between two gradient arrays relative to 1 + |a|.
    private func maxRelativeDifference(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        var worst: Float = 0
        for i in a.indices {
            worst = max(worst, abs(a[i] - b[i]) / (1 + abs(a[i])))
        }
        return worst
    }

    /// Pose sums must not change the parameter gradients. Two plain passes differ only by the float-atomic summation
    /// order of `rasterizeBackward`, so the sink run is compared with that run-to-run spread.
    @Test func poseSinkLeavesParameterGradientsUnchanged() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 120, seed: 9, shDegree: 1)
        let camera = rotatedCamera(width: 48, height: 48)
        let imageGradient = randomImageGradient(count: 48 * 48, seed: 40)
        let plain = try gpuBackward(context: context, cloud: cloud, camera: camera, settings: RasterSettings(),
                                    imageGradient: imageGradient, withSink: false)
        let plainAgain = try gpuBackward(context: context, cloud: cloud, camera: camera, settings: RasterSettings(),
                                         imageGradient: imageGradient, withSink: false)
        let withPose = try gpuBackward(context: context, cloud: cloud, camera: camera, settings: RasterSettings(),
                                       imageGradient: imageGradient, withSink: true)
        let spread = maxRelativeDifference(plain.gradients.sh, plainAgain.gradients.sh)
        let sinkDifference = max(
            maxRelativeDifference(plain.gradients.positions, withPose.gradients.positions),
            maxRelativeDifference(plain.gradients.logScales, withPose.gradients.logScales),
            maxRelativeDifference(plain.gradients.rotations, withPose.gradients.rotations),
            maxRelativeDifference(plain.gradients.opacityLogits, withPose.gradients.opacityLogits),
            maxRelativeDifference(plain.gradients.sh, withPose.gradients.sh))
        #expect(sinkDifference <= max(1e-6, 2 * spread), "sink changed parameter gradients by \(sinkDifference)")
    }

    /// No visible Gaussian: the sums stay zero and the pass dispatches nothing.
    @Test func noVisibleGaussiansGiveZeroPoseSums() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 40, seed: 3, shDegree: 1, depth: (-8)...(-2))
        let camera = makeTestCamera(width: 40, height: 30)
        let imageGradient = randomImageGradient(count: 40 * 30, seed: 41)
        let gpu = try gpuBackward(context: context, cloud: cloud, camera: camera, settings: RasterSettings(),
                                  imageGradient: imageGradient, withSink: true)
        #expect(gpu.sums.allSatisfy { $0 == 0 })
    }
}
