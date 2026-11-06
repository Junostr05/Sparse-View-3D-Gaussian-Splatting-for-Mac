import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// Interop with `Tools/synthetic/make_colmap_scene.py`. The fixtures are view 00 of a 64x48
/// scene written by the tool with seed 1 and two training views:
///
///     make_colmap_scene.py OUT --views 2 --size 64x48 --depth            -> synthetic_depth_metric_view_00.sdpt
///     make_colmap_scene.py OUT --views 2 --size 64x48 --relative-depth   -> synthetic_depth_relative_view_00.sdpt
///
/// Both files must have the same valid pixels, since they come from the same z-buffer.
/// Regenerate them with the same command if the tool or the codec changes.
@Suite struct SyntheticDepthFixtureTests {

    private static let width = 64
    private static let height = 48
    /// Pixel (32, 24) is the sphere centre. The camera ray hits the front surface at 2.5.
    private static let centreIndex = 24 * 64 + 32
    /// Valid (value > 0) pixels in the metric fixture, as counted by the tool.
    private static let validCount = 2342

    private func fixtureBytes(_ name: String) throws -> [UInt8] {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "sdpt",
                                                 subdirectory: "Fixtures"))
        return [UInt8](try Data(contentsOf: url))
    }

    private func load(_ name: String) throws -> DepthMap {
        let data = Data(try fixtureBytes(name))
        return try DepthMapCodec.decode(data)
    }

    private func littleEndianUInt32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(bytes[offset + index]) << UInt32(8 * index)
        }
        return value
    }

    @Test func metricFixtureHeaderMatchesTheTool() throws {
        let bytes = try fixtureBytes("synthetic_depth_metric_view_00")

        #expect(bytes.count == 24 + 64 * 48 * 4)                       // no confidence plane
        #expect(Array(bytes[0..<4]) == [0x53, 0x44, 0x50, 0x54])     // "SDPT"
        #expect(bytes[4] == 1 && bytes[5] == 0)                       // version 1
        #expect(bytes[6] == 0)                                        // kind metric
        #expect(bytes[7] == 0)                                        // flags: no confidence, not estimated
        #expect(littleEndianUInt32(bytes, at: 8) == 64)               // width
        #expect(littleEndianUInt32(bytes, at: 12) == 48)              // height
        for index in 16..<24 {
            #expect(bytes[index] == 0)                                // reserved
        }
    }

    @Test func metricFixtureDecodesWithImageSizeAndPlausibleValues() throws {
        let map = try load("synthetic_depth_metric_view_00")

        #expect(map.width == Self.width)
        #expect(map.height == Self.height)
        #expect(map.kind == .metric)
        #expect(map.confidence == nil)
        #expect(map.isEstimated == false)
        #expect(map.values.count == Self.width * Self.height)

        var valid = 0
        var minimum = Float.greatestFiniteMagnitude
        var maximum: Float = 0
        for value in map.values where value.isFinite && value > 0 {
            valid += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        }
        #expect(valid == Self.validCount)
        // Front of the sphere is 2.5 from the camera; the far side of the scene stays below 5.5.
        #expect(minimum >= 2.4 && maximum <= 5.5)

        let centre: Float = map.values[Self.centreIndex]
        #expect(abs(centre - 2.5) <= 0.1)
        // The top-left corner is background, so it is invalid.
        #expect(map.values[0] == 0)
    }

    @Test func relativeFixtureIsInverseDepthOnTheSameValidPixels() throws {
        let metric = try load("synthetic_depth_metric_view_00")
        let relative = try load("synthetic_depth_relative_view_00")

        #expect(relative.kind == .relativeInverseDepth)
        #expect(relative.width == Self.width && relative.height == Self.height)
        #expect(relative.values.count == metric.values.count)

        // value = a / z + b with a in [0.5, 2], b in [0, 0.5] and z in [2.5, 5.25] (the metric
        // range of this fixture), so every valid value lies in (0.09, 1.3].
        var sameValidity = true
        for index in 0..<metric.values.count {
            let metricValid = metric.values[index] > 0
            let relativeValid = relative.values[index] > 0
            if metricValid != relativeValid {
                sameValidity = false
            }
            if relativeValid {
                #expect(relative.values[index] > 0.09 && relative.values[index] <= 1.3)
            }
        }
        #expect(sameValidity)
    }
}
