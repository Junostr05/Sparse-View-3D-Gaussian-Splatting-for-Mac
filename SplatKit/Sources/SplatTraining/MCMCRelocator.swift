import Foundation
import Metal
import SplatCore
import SplatMetal

// MCMC relocation and growth (v1.2) at a wait point of the GPU trainer. The decision is `MCMCRelocation.plan`
// (SplatCore); this type applies it to the shared buffers on the CPU. The caller must have waited for the GPU.
// Positions, rotations and SH of a copy are the source's values. Opacity and scale are the relocated values.
// Adam moments of every modified Gaussian are reset to zero.

enum MCMCRelocator {
    /// Outcome of one relocation step.
    struct Result: Equatable {
        /// Dead Gaussians that were overwritten.
        var dead: Int
        /// Live Gaussians that were copied (each with copy count at least 2).
        var sources: Int
        /// Gaussians appended.
        var grown: Int
        /// True when the parameter and moment buffers were reallocated. Gradient buffers must be rebuilt.
        var grewCapacity: Bool
    }

    /// Relocates the dead Gaussians and appends the growth copies. Never raises the count above `maxGaussians`.
    static func run(parameters: GaussianBuffers, m: GaussianBuffers, v: GaussianBuffers,
                    settings: MCMCSettings, maxGaussians: Int, seed: UInt64) throws(MetalError) -> Result {
        let count = parameters.count
        precondition(m.count == count && v.count == count, "moments must have the parameters' count")
        let logits = readFloats(parameters.opacityLogits, count: count)
        let opacities: [Float] = logits.map { MCMCRelocation.sigmoid($0) }
        let growth = MCMCRelocation.growthCount(count: count, maxGaussians: maxGaussians,
                                                growthFraction: settings.growthFraction)
        let plan = MCMCRelocation.plan(opacities: opacities, deadOpacity: settings.deadOpacity,
                                       growth: growth, seed: seed)
        if plan.isEmpty {
            return Result(dead: 0, sources: 0, grown: 0, grewCapacity: false)
        }

        // Capacity for the appended Gaussians. Same 1.5x rule as the densifier, never above the budget.
        let target = count + plan.growthSources.count
        var grewCapacity = false
        if target > parameters.capacity {
            let grown = (3 * parameters.capacity + 1) / 2
            let newCapacity = max(target, min(maxGaussians, grown))
            try parameters.grow(toCapacity: newCapacity)
            try m.grow(toCapacity: newCapacity)
            try v.grow(toCapacity: newCapacity)
            grewCapacity = true
        }

        // Relocated opacity and scale of every source, computed from the values before any write.
        let logScales = pointer(parameters.logScales)
        var appearance: [Int: MCMCRelocatedAppearance] = [:]
        for entry in plan.copies {
            let base = 3 * entry.source
            let logScale = SIMD3<Float>(logScales[base], logScales[base + 1], logScales[base + 2])
            appearance[entry.source] = MCMCRelocation.relocated(logScale: logScale,
                                                                opacity: Double(opacities[entry.source]),
                                                                copies: entry.count)
        }
        let layout = GaussianBuffers.floatsPerGaussian(shDegree: parameters.shDegree)

        // Sources keep their position, rotation and SH, and take the relocated opacity and scale.
        for entry in plan.copies {
            guard let relocated = appearance[entry.source] else {
                preconditionFailure("missing relocation for source \(entry.source)")
            }
            write(relocated, to: entry.source, in: parameters)
            resetMoments(of: entry.source, layout: layout, m: m, v: v)
        }
        // Dead Gaussians become copies of their source.
        for replacement in plan.replacements {
            guard let relocated = appearance[replacement.source] else {
                preconditionFailure("missing relocation for source \(replacement.source)")
            }
            copyGaussian(from: replacement.source, to: replacement.target, in: parameters, layout: layout)
            write(relocated, to: replacement.target, in: parameters)
            resetMoments(of: replacement.target, layout: layout, m: m, v: v)
        }
        // Growth copies are appended after the existing Gaussians.
        for (offset, source) in plan.growthSources.enumerated() {
            guard let relocated = appearance[source] else {
                preconditionFailure("missing relocation for source \(source)")
            }
            let target = count + offset
            copyGaussian(from: source, to: target, in: parameters, layout: layout)
            write(relocated, to: target, in: parameters)
            resetMoments(of: target, layout: layout, m: m, v: v)
        }

        parameters.count = target
        m.count = target
        v.count = target
        return Result(dead: plan.replacements.count, sources: plan.copies.count,
                      grown: plan.growthSources.count, grewCapacity: grewCapacity)
    }

    // MARK: - Buffer access (the GPU must be idle)

    private static func pointer(_ buffer: MTLBuffer) -> UnsafeMutablePointer<Float> {
        buffer.contents().assumingMemoryBound(to: Float.self)
    }

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: pointer(buffer), count: count))
    }

    /// Writes the relocated opacity logit and log-scales of one Gaussian.
    private static func write(_ relocated: MCMCRelocatedAppearance, to index: Int, in store: GaussianBuffers) {
        pointer(store.opacityLogits)[index] = relocated.opacityLogit
        let base = 3 * index
        let logScales = pointer(store.logScales)
        logScales[base] = relocated.logScale.x
        logScales[base + 1] = relocated.logScale.y
        logScales[base + 2] = relocated.logScale.z
    }

    /// Copies positions, rotations and SH from `source` to `target`. Opacity and scale are written separately.
    /// `layout` is `floatsPerGaussian`: [3, 3, 4, 1, 3K], so the groups are 0 positions, 2 rotations, 4 SH.
    private static func copyGaussian(from source: Int, to target: Int, in store: GaussianBuffers, layout: [Int]) {
        let groups = store.groups
        for group in [0, 2, 4] {
            let floats = layout[group]
            let base = pointer(groups[group])
            (base + target * floats).update(from: base + source * floats, count: floats)
        }
    }

    /// Zeroes the Adam moments of one Gaussian in every group.
    private static func resetMoments(of index: Int, layout: [Int], m: GaussianBuffers, v: GaussianBuffers) {
        for store in [m, v] {
            let groups = store.groups
            for group in 0..<groups.count {
                let floats = layout[group]
                (pointer(groups[group]) + index * floats).update(repeating: 0, count: floats)
            }
        }
    }
}
