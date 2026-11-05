// swift-tools-version: 6.2
import PackageDescription

var products: [Product] = [
    .library(name: "SplatCore", targets: ["SplatCore"]),
    .library(name: "SplatIO", targets: ["SplatIO"]),
]

var targets: [Target] = [
    .target(name: "SplatCore"),
    .target(name: "SplatIO", dependencies: ["SplatCore"]),
    .testTarget(name: "SplatCoreTests", dependencies: ["SplatCore"]),
    .testTarget(name: "SplatIOTests", dependencies: ["SplatIO"]),
]

#if os(macOS)
products.append(.library(name: "SplatMetal", targets: ["SplatMetal"]))
targets += [
    .target(
        name: "SplatMetal",
        dependencies: ["SplatCore"],
        resources: [.copy("Shaders")]
    ),
    .testTarget(name: "SplatMetalTests", dependencies: ["SplatMetal", "SplatCore"]),
]
#endif

let package = Package(
    name: "SplatKit",
    platforms: [.macOS(.v26)],
    products: products,
    targets: targets
)
