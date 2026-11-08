import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// A degree-0 cloud whose Gaussians have distinct positions and varied scales and opacities.
private func webCloud(count: Int) throws -> GaussianCloud {
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    for i in 0..<count {
        let t: Float = Float(i)
        let x: Float = 0.5 * t - 4
        let y: Float = 0.25 * t
        let z: Float = -t
        positions += [x, y, z]
        let ls0: Float = -2 + 0.05 * Float(i % 17)
        let ls1: Float = -1.5 + 0.03 * Float((i * 7) % 11)
        let ls2: Float = -3 + 0.02 * Float((i * 3) % 13)
        logScales += [ls0, ls1, ls2]
        rotations += [1, 0, 0, 0]
        opacity.append(-1 + 0.04 * Float((i * 5) % 37))
        sh += [0.1 * Float(i % 5), 0.05, -0.1]
    }
    return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                             rotations: rotations, opacityLogits: opacity, sh: sh)
}

/// The text between `openingTag` and the next closing script tag.
private func scriptBlock(_ html: String, openingTag: String) -> String? {
    guard let open = html.range(of: openingTag),
          let close = html[open.upperBound...].range(of: "</script>") else {
        return nil
    }
    return String(html[open.upperBound..<close.lowerBound])
}

/// The numbers of a JSON array, whether Foundation parsed them as Double or Int.
private func numbers(_ value: Any?) -> [Double]? {
    guard let array = value as? [Any] else { return nil }
    var result: [Double] = []
    for element in array {
        if let number = element as? Double {
            result.append(number)
        } else if let integer = element as? Int {
            result.append(Double(integer))
        } else {
            return nil
        }
    }
    return result
}

private func decodeBase64Gzip(_ text: String) throws -> [UInt8] {
    guard let compressed = Data(base64Encoded: text) else {
        throw SplatIOError.corrupt("base64")
    }
    return try Gzip.decompress(compressed)
}

private func float32(_ bytes: [UInt8], at offset: Int) -> Float {
    let bits: UInt32 = UInt32(bytes[offset])
        | UInt32(bytes[offset + 1]) << 8
        | UInt32(bytes[offset + 2]) << 16
        | UInt32(bytes[offset + 3]) << 24
    return Float(bitPattern: bits)
}

@Test func pageHoldsOneDataBlockThatDecodesToTheSplatRecords() throws {
    let cloud = try webCloud(count: 40)
    let result = try WebViewerExporter.export(cloud: cloud)
    let html = String(decoding: result.data, as: UTF8.self)

    #expect(html.hasPrefix("<!DOCTYPE html>"))
    #expect(html.components(separatedBy: "id=\"data\"").count == 2)
    #expect(result.includedCount == 40)
    #expect(result.droppedCount == 0)
    #expect(result.removedNonFiniteCount == 0)

    let encoded = try #require(scriptBlock(html, openingTag: "<script id=\"data\" type=\"application/octet-stream\">"))
    let records = try decodeBase64Gzip(encoded)
    #expect(records.count == 40 * 32)
    #expect(records == [UInt8](SplatWriter.write(cloud)))

    // Small clouds also carry the uncompressed fallback, with the same records.
    let raw = try #require(scriptBlock(html, openingTag: "<script id=\"data-raw\" type=\"application/octet-stream\">"))
    #expect(Data(base64Encoded: raw) == Data(records))
}

@Test func truncationKeepsTheMostImportantGaussians() throws {
    let count = 100
    let cloud = try webCloud(count: count)
    var options = WebViewerOptions()
    options.maxGaussians = 30
    let result = try WebViewerExporter.export(cloud: cloud, options: options)
    #expect(result.includedCount == 30)
    #expect(result.droppedCount == 70)

    // Importance is exp(sum of log scales) × opacity, computed here independently of the exporter.
    var keys: [Double] = []
    for i in 0..<count {
        let scaleSum: Double = Double(cloud.logScales[3 * i]) + Double(cloud.logScales[3 * i + 1])
            + Double(cloud.logScales[3 * i + 2])
        let opacity: Double = 1 / (1 + exp(-Double(cloud.opacityLogits[i])))
        keys.append(exp(scaleSum) * opacity)
    }
    var order: [Int] = Array(0..<count)
    order.sort { (a: Int, b: Int) -> Bool in
        if keys[a] != keys[b] {
            return keys[a] > keys[b]
        }
        return a < b
    }
    var expected: Set<Int> = []
    for index in order.prefix(30) {
        expected.insert(index)
    }

    let html = String(decoding: result.data, as: UTF8.self)
    let encoded = try #require(scriptBlock(html, openingTag: "<script id=\"data\" type=\"application/octet-stream\">"))
    let records = try decodeBase64Gzip(encoded)
    #expect(records.count == 30 * 32)

    var indexByPosition: [SIMD3<Float>: Int] = [:]
    for i in 0..<count {
        let position: SIMD3<Float> = SIMD3(cloud.positions[3 * i], cloud.positions[3 * i + 1], cloud.positions[3 * i + 2])
        indexByPosition[position] = i
    }
    var included: Set<Int> = []
    for record in 0..<30 {
        let offset: Int = record * 32
        let position: SIMD3<Float> = SIMD3(float32(records, at: offset), float32(records, at: offset + 4),
                                           float32(records, at: offset + 8))
        if let index = indexByPosition[position] {
            included.insert(index)
        }
    }
    #expect(included == expected)
}

@Test func pageHasNoExternalReferences() throws {
    let result = try WebViewerExporter.export(cloud: try webCloud(count: 12))
    let html = String(decoding: result.data, as: UTF8.self)
    #expect(!html.contains("http://"))
    #expect(!html.contains("https://"))
    #expect(!html.contains("src="))
    #expect(!html.contains("<link"))
    #expect(!html.contains("%%"))
}

@Test func initialCameraIsEmbedded() throws {
    var options = WebViewerOptions()
    options.camera = WebViewerCamera(position: SIMD3(1.25, -2.5, 3.75), target: SIMD3(0.5, 0.25, -1),
                                     up: SIMD3(0, -1, 0), fovY: 0.9)
    options.title = "Test scene"
    options.background = SIMD3(0.25, 0.5, 1)
    let result = try WebViewerExporter.export(cloud: try webCloud(count: 5), options: options)
    let html = String(decoding: result.data, as: UTF8.self)

    let json = try #require(scriptBlock(html, openingTag: "<script id=\"config\" type=\"application/json\">"))
    let object = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    #expect(object["title"] as? String == "Test scene")
    #expect(object["count"] as? Int == 5)
    #expect(object["total"] as? Int == 5)
    #expect(object["upAxis"] as? String == "negativeY")

    let camera = try #require(object["camera"] as? [String: Any])
    #expect(numbers(camera["position"]) == [1.25, -2.5, 3.75])
    #expect(numbers(camera["target"]) == [0.5, 0.25, -1])
    let fov = try #require(camera["fovY"] as? Double)
    #expect(abs(fov - 0.9) < 1e-6)
    #expect(numbers(object["up"]) == [0, -1, 0])
    #expect(numbers(object["background"]) == [0.25, 0.5, 1])
    #expect(html.contains("<title>Test scene</title>"))
}

@Test func nonFiniteGaussiansAreRemovedAndCounted() throws {
    let base = try webCloud(count: 6)
    var positions = base.positions
    positions[3] = Float.nan
    let cloud = try GaussianCloud(shDegree: 0, positions: positions, logScales: base.logScales,
                                  rotations: base.rotations, opacityLogits: base.opacityLogits, sh: base.sh)
    let result = try WebViewerExporter.export(cloud: cloud)
    #expect(result.removedNonFiniteCount == 1)
    #expect(result.includedCount == 5)
}

@Test func emptyCloudThrows() throws {
    let cloud = try GaussianCloud(shDegree: 0, positions: [], logScales: [], rotations: [],
                                  opacityLogits: [], sh: [])
    #expect(throws: SplatIOError.empty) {
        _ = try WebViewerExporter.export(cloud: cloud)
    }
}

@Test func titleIsEscapedAndCannotInjectPlaceholders() throws {
    var options = WebViewerOptions()
    options.title = "A </script><b>&\"x\" %%DATA%%"
    let result = try WebViewerExporter.export(cloud: try webCloud(count: 3), options: options)
    let html = String(decoding: result.data, as: UTF8.self)
    // The title is escaped, and it is inserted last, so the text that looks like a placeholder stays literal.
    #expect(html.contains("<title>A &lt;/script&gt;&lt;b&gt;&amp;&quot;x&quot; %%DATA%%</title>"))
    // The four script elements: config, data, raw data and the page script. The title did not close any of them.
    #expect(html.components(separatedBy: "</script>").count == 5)
    let json = try #require(scriptBlock(html, openingTag: "<script id=\"config\" type=\"application/json\">"))
    let object = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    #expect(object["title"] as? String == "A </script><b>&\"x\" %%DATA%%")
}

@Test func estimatedSizeFollowsTheFormula() {
    #expect(WebViewerExporter.estimatedByteCount(gaussianCount: 1_000_000) == 25_600_000)
    #expect(WebViewerExporter.estimatedByteCount(gaussianCount: 0) == 0)
    #expect(WebViewerOptions.defaultMaxGaussians == 1_000_000)
}
