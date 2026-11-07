import Foundation
import SplatIO
import Testing

@Suite struct DepthModelCatalogTests {

    @Test func mogeCatalogEntryIsConfiguredWithPinnedPackage() {
        let descriptor = ModelCatalog.moge2SmallInt8
        #expect(descriptor.id == "moge-2-vits-int8")
        #expect(descriptor.displayName == "MoGe-2 Small (depth)")
        #expect(descriptor.version == "e01dfc7")
        #expect(descriptor.license == "MIT")
        #expect(descriptor.baseURL == URL(string: "https://huggingface.co/ebowwa/moge-2-vits-normal-coreml/resolve/e01dfc7e5148e57fbad3b72fee832f8e9d6fdbda"))
        #expect(descriptor.packageName == "moge_vits_normal_int8.mlpackage")
        #expect(descriptor.files.map(\.relativePath) == [
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin",
        ])
        #expect(descriptor.files.map(\.sizeBytes) == [617, 301_577, 36_128_704])
        #expect(descriptor.files.map(\.sha256) == [
            "a370d9b38cca877bb62ba44ab37d94f751358acd146fbf7a8f7ec23f72ff8e16",
            "9a117c8978535ea9d9d02d1aa8031cc85b1d5f0e3acb75ee59eb416023bfd937",
            "b3a01f24323632b5a4e6d41dfefadb69cad505efeb399c736ca0e8c0817cde2b",
        ])
        let expectedTotal: Int64 = 617 + 301_577 + 36_128_704
        #expect(descriptor.totalSizeBytes == expectedTotal)
        #expect(descriptor.isConfigured)
        #expect(ModelCatalog.all.contains(descriptor))
    }

    @Test func catalogListsEachDepthModelOnceAndConfiguresAll() {
        #expect(ModelCatalog.all.map(\.id) == ["depth-anything-v2-small", "moge-2-vits-int8"])
        #expect(ModelCatalog.all.allSatisfy { $0.isConfigured })
    }
}
