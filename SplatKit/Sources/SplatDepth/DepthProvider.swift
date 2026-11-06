import Foundation
import SplatCore

/// Source of a depth map for one training view (Phase 4 P1).
///
/// Returning `nil` means "no depth for this view". A thrown error is reported per view by
/// `DepthPreparation`, never propagated to the caller.
public protocol DepthProvider: Sendable {
    func depthMap(for view: TrainingView) throws -> DepthMap?
}

/// Depth maps supplied with the project (LiDAR captures, precomputed `.sdpt` files), keyed by
/// view name.
public struct SuppliedDepthProvider: DepthProvider {
    private let maps: [String: DepthMap]

    public init(maps: [String: DepthMap]) {
        self.maps = maps
    }

    public func depthMap(for view: TrainingView) throws -> DepthMap? {
        maps[view.name]
    }
}

/// Asks each provider in order and returns the first map found. A provider that throws is
/// skipped, so a broken source never hides a working one.
public struct FirstAvailableDepthProvider: DepthProvider {
    private let providers: [any DepthProvider]

    public init(_ providers: [any DepthProvider]) {
        self.providers = providers
    }

    public func depthMap(for view: TrainingView) throws -> DepthMap? {
        for provider in providers {
            if let map = try? provider.depthMap(for: view) {
                return map
            }
        }
        return nil
    }
}
