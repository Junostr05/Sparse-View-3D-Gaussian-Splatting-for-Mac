import Foundation
import Testing
import SplatCore
import SplatIO

/// Flat index into `GaussianCloud.sh` for Gaussian `i`, SH coefficient `k` (K = 16), channel `c`.
private func shIndex(gaussian i: Int, coefficient k: Int, channel c: Int, K: Int = 16) -> Int {
    (i * K + k) * 3 + c
}

@Test func readsDegree0() throws {
    let props = standardProperties(restCount: 0)
    let data = makePLY(properties: props, rows: [
        makeRow(props, ["x": 1, "y": 2, "z": 3,
                        "f_dc_0": 0.1, "f_dc_1": 0.2, "f_dc_2": 0.3,
                        "opacity": 0.5,
                        "scale_0": -1, "scale_1": -2, "scale_2": -3,
                        "rot_0": 1, "rot_1": 0, "rot_2": 0, "rot_3": 0]),
        makeRow(props, ["x": 4, "y": 5, "z": 6,
                        "f_dc_0": 0.4, "f_dc_1": 0.5, "f_dc_2": 0.6,
                        "opacity": -0.5,
                        "scale_0": -4, "scale_1": -5, "scale_2": -6,
                        "rot_0": 0, "rot_1": 1, "rot_2": 0, "rot_3": 0]),
    ])

    let cloud = try PLYReader.read(data)

    #expect(cloud.shDegree == 0)
    #expect(cloud.count == 2)
    #expect(approxEqual(cloud.positions, [1, 2, 3, 4, 5, 6], tol: 1e-6))
    #expect(approxEqual(cloud.logScales, [-1, -2, -3, -4, -5, -6], tol: 1e-6))
    #expect(approxEqual(cloud.rotations, [1, 0, 0, 0, 0, 1, 0, 0], tol: 1e-6))
    #expect(approxEqual(cloud.opacityLogits, [0.5, -0.5], tol: 1e-6))
    #expect(approxEqual(cloud.sh, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], tol: 1e-6))
}

@Test func readsDegree3RestLayout() throws {
    let props = standardProperties(restCount: 45)
    var values: [String: Double] = ["rot_0": 1]
    for i in 0..<45 {
        values["f_rest_\(i)"] = Double(i)
    }
    let data = makePLY(properties: props, rows: [makeRow(props, values)])

    let cloud = try PLYReader.read(data)

    #expect(cloud.shDegree == 3)
    #expect(cloud.sh.count == 1 * 16 * 3)
    // f_rest is stored channel-major: f_rest[c * 15 + (k - 1)].
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 1, channel: 0)] == 0)
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 1, channel: 1)] == 15)
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 1, channel: 2)] == 30)
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 15, channel: 2)] == 44)
}

@Test func readsReorderedAndExtraProperties() throws {
    // Normals first, x stored as double, f_rest after opacity, an unused uchar property.
    let props: [(name: String, type: String)] = [
        (name: "nx", type: "float"), (name: "ny", type: "float"), (name: "nz", type: "float"),
        (name: "x", type: "double"), (name: "y", type: "float"), (name: "z", type: "float"),
        (name: "f_dc_0", type: "float"), (name: "f_dc_1", type: "float"), (name: "f_dc_2", type: "float"),
        (name: "opacity", type: "float"),
        (name: "f_rest_0", type: "float"), (name: "f_rest_1", type: "float"), (name: "f_rest_2", type: "float"),
        (name: "f_rest_3", type: "float"), (name: "f_rest_4", type: "float"), (name: "f_rest_5", type: "float"),
        (name: "f_rest_6", type: "float"), (name: "f_rest_7", type: "float"), (name: "f_rest_8", type: "float"),
        (name: "scale_0", type: "float"), (name: "scale_1", type: "float"), (name: "scale_2", type: "float"),
        (name: "rot_0", type: "float"), (name: "rot_1", type: "float"), (name: "rot_2", type: "float"),
        (name: "rot_3", type: "float"),
        (name: "label", type: "uchar"),
    ]
    let data = makePLY(properties: props, rows: [makeRow(props, [
        "nx": 0, "ny": 0, "nz": 1,
        "x": 1.25, "y": -2, "z": 3,
        "f_dc_0": 0.1, "f_dc_1": 0.2, "f_dc_2": 0.3,
        "opacity": 0.75,
        "f_rest_0": 0, "f_rest_1": 1, "f_rest_2": 2, "f_rest_3": 3, "f_rest_4": 4,
        "f_rest_5": 5, "f_rest_6": 6, "f_rest_7": 7, "f_rest_8": 8,
        "scale_0": -1, "scale_1": -1, "scale_2": -1,
        "rot_0": 1,
        "label": 7,
    ])])

    let cloud = try PLYReader.read(data)

    #expect(cloud.shDegree == 1)
    #expect(approxEqual(cloud.positions, [1.25, -2, 3], tol: 1e-6))
    #expect(approxEqual(cloud.logScales, [-1, -1, -1], tol: 1e-6))
    #expect(approxEqual(cloud.rotations, [1, 0, 0, 0], tol: 1e-6))
    #expect(approxEqual(cloud.opacityLogits, [0.75], tol: 1e-6))
    // Degree 1: K = 4 coefficients, f_rest is channel-major with 3 coefficients per channel.
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 0, channel: 0, K: 4)] == 0.1)
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 1, channel: 0, K: 4)] == 0)
    #expect(cloud.sh[shIndex(gaussian: 0, coefficient: 3, channel: 2, K: 4)] == 8)
}

@Test func acceptsCRLFHeader() throws {
    let props = standardProperties(restCount: 0)
    let data = makePLY(properties: props, rows: [makeRow(props, ["x": 7, "y": 8, "z": 9, "rot_0": 1])],
                       lineEnding: "\r\n")

    let cloud = try PLYReader.read(data)

    #expect(cloud.count == 1)
    #expect(approxEqual(cloud.positions, [7, 8, 9], tol: 1e-6))
}

@Test func rejectsASCIIFormat() {
    let props = standardProperties(restCount: 0)
    #expect(throws: SplatIOError.unsupportedFormat("ascii")) {
        try PLYReader.read(makePLY(format: "ascii 1.0", properties: props, rows: []))
    }
}

@Test func rejectsBigEndianFormat() {
    let props = standardProperties(restCount: 0)
    #expect(throws: SplatIOError.unsupportedFormat("binary_big_endian")) {
        try PLYReader.read(makePLY(format: "binary_big_endian 1.0", properties: props, rows: []))
    }
}

@Test func rejectsMissingOpacity() {
    let props = standardProperties(restCount: 0).filter { $0.name != "opacity" }
    let data = makePLY(properties: props, rows: [makeRow(props, [:])])
    #expect(throws: SplatIOError.missingProperty("opacity")) {
        try PLYReader.read(data)
    }
}

@Test func rejectsBadRestCount() {
    let props = standardProperties(restCount: 10)
    let data = makePLY(properties: props, rows: [makeRow(props, [:])])
    #expect(throws: SplatIOError.unsupportedSHCount(10)) {
        try PLYReader.read(data)
    }
}

@Test func rejectsTruncatedBody() {
    let props = standardProperties(restCount: 0)
    let stride = props.count * 4
    // Header declares 2 vertices, body holds only 1.
    let data = makePLY(properties: props, rows: [makeRow(props, ["rot_0": 1])], vertexCount: 2)
    #expect(throws: SplatIOError.truncated(expected: 2 * stride, actual: stride)) {
        try PLYReader.read(data)
    }
}

@Test func rejectsZeroVertices() {
    let props = standardProperties(restCount: 0)
    #expect(throws: SplatIOError.empty) {
        try PLYReader.read(makePLY(properties: props, rows: []))
    }
}

@Test func rejectsNonPLY() {
    #expect(throws: SplatIOError.notPLY) {
        try PLYReader.read(Data("hello".utf8))
    }
}

@Test func rejectsHeaderWithoutEndHeader() {
    let data = Data("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\n".utf8)
    #expect(throws: SplatIOError.notPLY) {
        try PLYReader.read(data)
    }
}

@Test func rejectsListPropertyInVertex() {
    // Hand-built header: makePLY only writes scalar properties.
    let header = Data("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty list uchar int idx\nend_header\n".utf8)
    #expect(throws: SplatIOError.unsupportedFormat("list property idx")) {
        try PLYReader.read(header)
    }
}

@Test func rejectsVertexNotFirstElement() {
    let header = Data("ply\nformat binary_little_endian 1.0\nelement face 1\nproperty float x\nelement vertex 1\nproperty float x\nend_header\n".utf8)
    #expect(throws: SplatIOError.unsupportedFormat("first element must be vertex")) {
        try PLYReader.read(header)
    }
}
