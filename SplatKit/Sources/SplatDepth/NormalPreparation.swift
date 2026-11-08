import Foundation
import SplatCore

/// Source of camera-frame surface normals for one training view (v1.5 normal prior). Unit normals in the OpenCV
/// frame pointing towards the camera, `camera.width × camera.height` entries, zero where there is no estimate.
public protocol NormalProvider: Sendable {
    func normalMap(for view: TrainingView) throws -> [SIMD3<Float>]?
}

extension MoGeDepthEstimator: NormalProvider {}

/// Attaches normal priors to the depth priors of a dataset (v1.5). Runs after `DepthPreparation`.
public enum NormalPreparation {
    /// Sets `depth.normals` of every view that has a depth prior and a usable normal map. Views without a depth
    /// prior get none (the normal term needs the depth pass). Provider failures and maps of the wrong size are
    /// skipped and counted, never thrown.
    public static func attach(_ dataset: TrainingDataset,
                              provider: any NormalProvider) -> (dataset: TrainingDataset, attached: Int, failed: Int) {
        var result: TrainingDataset = dataset
        var attached: Int = 0
        var failed: Int = 0
        for index in 0..<result.views.count {
            guard let prior = result.views[index].depth else { continue }
            let view: TrainingView = result.views[index]
            let normals: [SIMD3<Float>]?
            do {
                normals = try provider.normalMap(for: view)
            } catch {
                failed += 1
                continue
            }
            guard let normals, normals.count == prior.width * prior.height else {
                failed += 1
                continue
            }
            result.views[index].depth?.normals = normals
            attached += 1
        }
        return (result, attached, failed)
    }
}
