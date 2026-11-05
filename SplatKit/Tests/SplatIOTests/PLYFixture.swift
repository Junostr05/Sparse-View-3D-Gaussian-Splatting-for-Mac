import Foundation

/// Test-only builder for binary little-endian 3DGS PLY files.
///
/// `rows[i]` holds one value per entry of `properties`, in declaration order.
/// Values are written in each property's declared scalar type, little-endian.
func makePLY(format: String = "binary_little_endian 1.0",
             properties: [(name: String, type: String)],
             rows: [[Double]],
             lineEnding: String = "\n",
             vertexCount: Int? = nil) -> Data {
    var header = "ply\(lineEnding)"
    header += "format \(format)\(lineEnding)"
    header += "comment test fixture\(lineEnding)"
    header += "element vertex \(vertexCount ?? rows.count)\(lineEnding)"
    for property in properties {
        header += "property \(property.type) \(property.name)\(lineEnding)"
    }
    header += "end_header\(lineEnding)"

    var data = Data(header.utf8)
    for row in rows {
        precondition(row.count == properties.count, "row width must match property count")
        for (value, property) in zip(row, properties) {
            appendScalar(value, type: property.type, to: &data)
        }
    }
    return data
}

/// Position, color, opacity, scale and rotation properties followed by `restCount` `f_rest_*`
/// coefficients. All properties are `float`.
func standardProperties(restCount: Int) -> [(name: String, type: String)] {
    var result: [(name: String, type: String)] = []
    for name in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
        result.append((name: name, type: "float"))
    }
    for i in 0..<restCount {
        result.append((name: "f_rest_\(i)", type: "float"))
    }
    return result
}

/// Builds one row by looking up each property name in `values`; missing names become 0.
func makeRow(_ properties: [(name: String, type: String)], _ values: [String: Double]) -> [Double] {
    properties.map { values[$0.name] ?? 0 }
}

private func appendScalar(_ value: Double, type: String, to data: inout Data) {
    switch type {
    case "char", "int8":
        appendLittleEndian(Int8(value), to: &data)
    case "uchar", "uint8":
        appendLittleEndian(UInt8(value), to: &data)
    case "short", "int16":
        appendLittleEndian(Int16(value), to: &data)
    case "ushort", "uint16":
        appendLittleEndian(UInt16(value), to: &data)
    case "int", "int32":
        appendLittleEndian(Int32(value), to: &data)
    case "uint", "uint32":
        appendLittleEndian(UInt32(value), to: &data)
    case "float", "float32":
        appendLittleEndian(Float(value).bitPattern, to: &data)
    case "double", "float64":
        appendLittleEndian(value.bitPattern, to: &data)
    default:
        preconditionFailure("unknown PLY scalar type \(type)")
    }
}

private func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}
