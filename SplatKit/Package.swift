// swift-tools-version: 6.2
import PackageDescription

var products: [Product] = [
    .library(name: "SplatCore", targets: ["SplatCore"]),
    .library(name: "SplatIO", targets: ["SplatIO"]),
]

var targets: [Target] = [
    .target(name: "SplatCore"),
    .target(name: "SplatIO", dependencies: ["SplatCore"]),
    .testTarget(name: "SplatCoreTests", dependencies: ["SplatCore"], resources: [.copy("Fixtures")]),
    .testTarget(name: "SplatIOTests", dependencies: ["SplatIO"], resources: [.copy("Fixtures")]),
]

#if os(macOS)
products.append(.library(name: "SplatMetal", targets: ["SplatMetal"]))
products.append(.library(name: "SplatTraining", targets: ["SplatTraining"]))
products.append(.library(name: "SplatDepth", targets: ["SplatDepth"]))
targets += [
    .target(
        name: "SplatMetal",
        dependencies: ["SplatCore"],
        resources: [.copy("Shaders")]
    ),
    .testTarget(
        name: "SplatMetalTests",
        dependencies: ["SplatMetal", "SplatCore"],
        resources: [.copy("Fixtures")]
    ),
    // Phase 3 Task 15. Nonisolated default isolation: never set defaultIsolation(MainActor.self) here.
    .target(
        name: "SplatTraining",
        dependencies: ["SplatCore", "SplatMetal"]
    ),
    .testTarget(
        name: "SplatTrainingTests",
        dependencies: ["SplatTraining", "SplatMetal", "SplatCore"]
    ),
    // Phase 4 Task 5: depth alignment, providers and preparation. Foundation only for now; the
    // Core ML depth model is added here by a later task. Nonisolated default isolation.
    .target(
        name: "SplatDepth",
        dependencies: ["SplatCore"]
    ),
    .testTarget(
        name: "SplatDepthTests",
        dependencies: ["SplatDepth", "SplatCore"]
    ),
]
#endif

let package = Package(
    name: "SplatKit",
    platforms: [.macOS(.v26)],
    products: products,
    targets: targets
)
