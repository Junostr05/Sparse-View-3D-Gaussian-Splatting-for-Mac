import Foundation
import SplatCore

/// Reads binary little-endian 3D Gaussian Splatting PLY files.
public enum PLYReader {
    public static func read(_ data: Data) throws(SplatIOError) -> GaussianCloud {
        let result: Result<GaussianCloud, SplatIOError> = data.withUnsafeBytes { raw in
            do throws(SplatIOError) {
                return .success(try parse(raw))
            } catch {
                return .failure(error)
            }
        }
        return try result.get()
    }

    // MARK: - Header

    private enum PropertyKind {
        case scalar(ScalarType)
        case list
    }

    private struct PropertyDecl {
        let name: String
        let kind: PropertyKind
    }

    private struct ElementDecl {
        let name: String
        let count: Int
        var properties: [PropertyDecl] = []
    }

    private struct Header {
        let elements: [ElementDecl]
        let formatName: String
        /// Byte offset of the first byte after `end_header` and its line ending.
        let bodyStart: Int
    }

    private static func parseHeader(_ raw: UnsafeRawBufferPointer) throws(SplatIOError) -> Header {
        var elements: [ElementDecl] = []
        var formatName: String?
        var bodyStart: Int?
        var lineStart = 0
        var sawMagic = false

        header: while lineStart < raw.count {
            guard let newline = raw[lineStart...].firstIndex(of: 0x0A) else { break }
            var contentEnd = newline
            if contentEnd > lineStart && raw[contentEnd - 1] == 0x0D {
                contentEnd -= 1
            }
            let line = String(decoding: raw[lineStart..<contentEnd], as: UTF8.self)
            lineStart = newline + 1

            if !sawMagic {
                guard line == "ply" else { throw .notPLY }
                sawMagic = true
                continue
            }

            let tokens = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard let keyword = tokens.first else { continue }
            switch keyword {
            case "format":
                guard tokens.count >= 2 else { throw .notPLY }
                formatName = String(tokens[1])
            case "element":
                guard tokens.count == 3, let count = Int(tokens[2]) else { throw .notPLY }
                elements.append(ElementDecl(name: String(tokens[1]), count: count))
            case "property":
                guard !elements.isEmpty, tokens.count >= 3 else { throw .notPLY }
                let kind: PropertyKind
                let name: String
                if tokens[1] == "list" {
                    // property list <count type> <item type> <name>
                    guard tokens.count == 5 else { throw .notPLY }
                    kind = .list
                    name = String(tokens[4])
                } else {
                    guard tokens.count == 3 else { throw .notPLY }
                    guard let type = ScalarType(name: String(tokens[1])) else {
                        throw .unsupportedFormat(String(tokens[1]))
                    }
                    kind = .scalar(type)
                    name = String(tokens[2])
                }
                elements[elements.count - 1].properties.append(PropertyDecl(name: name, kind: kind))
            case "end_header":
                bodyStart = lineStart
                break header
            default:
                // `comment`, `obj_info` and unknown keywords are ignored.
                continue
            }
        }

        guard let bodyStart, let formatName else { throw .notPLY }
        return Header(elements: elements, formatName: formatName, bodyStart: bodyStart)
    }

    // MARK: - Body

    /// Byte offset of a property within one vertex row, and its scalar type.
    private struct Field {
        let offset: Int
        let type: ScalarType

        @inline(__always)
        func load(_ raw: UnsafeRawBufferPointer, rowStart: Int) -> Float {
            type.read(raw, at: rowStart + offset)
        }
    }

    private static func parse(_ raw: UnsafeRawBufferPointer) throws(SplatIOError) -> GaussianCloud {
        let header = try parseHeader(raw)

        guard header.formatName == "binary_little_endian" else {
            throw .unsupportedFormat(header.formatName)
        }
        guard let vertex = header.elements.first, vertex.name == "vertex" else {
            throw .unsupportedFormat("first element must be vertex")
        }

        var fieldByName: [String: Field] = [:]
        var stride = 0
        for property in vertex.properties {
            switch property.kind {
            case .list:
                throw .unsupportedFormat("list property \(property.name)")
            case .scalar(let type):
                fieldByName[property.name] = Field(offset: stride, type: type)
                stride += type.size
            }
        }

        func field(_ name: String) throws(SplatIOError) -> Field {
            guard let field = fieldByName[name] else { throw .missingProperty(name) }
            return field
        }
        func fields(_ names: [String]) throws(SplatIOError) -> [Field] {
            var result: [Field] = []
            result.reserveCapacity(names.count)
            for name in names {
                result.append(try field(name))
            }
            return result
        }

        let positionFields = try fields(["x", "y", "z"])
        let dcFields = try fields(["f_dc_0", "f_dc_1", "f_dc_2"])
        let opacityField = try field("opacity")
        let scaleFields = try fields(["scale_0", "scale_1", "scale_2"])
        let rotationFields = try fields(["rot_0", "rot_1", "rot_2", "rot_3"])

        let restCount = vertex.properties.lazy.filter { $0.name.hasPrefix("f_rest_") }.count
        let shDegree: Int
        switch restCount {
        case 0: shDegree = 0
        case 9: shDegree = 1
        case 24: shDegree = 2
        case 45: shDegree = 3
        default: throw .unsupportedSHCount(restCount)
        }
        let K = GaussianCloud.coefficientCount(degree: shDegree)
        // f_rest is channel-major: f_rest[c * (K - 1) + (k - 1)].
        let restFields = try fields((0..<restCount).map { "f_rest_\($0)" })

        let n = vertex.count
        guard n >= 0 else { throw .invalidSize(n) }
        guard n > 0 else { throw .empty }

        let (expected, overflow) = n.multipliedReportingOverflow(by: stride)
        guard !overflow else { throw .invalidSize(n) }
        let available = raw.count - header.bodyStart
        guard available >= expected else {
            throw .truncated(expected: expected, actual: available)
        }

        var positions = [Float](repeating: 0, count: 3 * n)
        var logScales = [Float](repeating: 0, count: 3 * n)
        var rotations = [Float](repeating: 0, count: 4 * n)
        var opacityLogits = [Float](repeating: 0, count: n)
        var sh = [Float](repeating: 0, count: n * K * 3)

        for i in 0..<n {
            let rowStart = header.bodyStart + i * stride
            for c in 0..<3 {
                positions[3 * i + c] = positionFields[c].load(raw, rowStart: rowStart)
                logScales[3 * i + c] = scaleFields[c].load(raw, rowStart: rowStart)
                sh[(i * K) * 3 + c] = dcFields[c].load(raw, rowStart: rowStart)
            }
            for q in 0..<4 {
                rotations[4 * i + q] = rotationFields[q].load(raw, rowStart: rowStart)
            }
            opacityLogits[i] = opacityField.load(raw, rowStart: rowStart)
            for k in 1..<K {
                for c in 0..<3 {
                    sh[(i * K + k) * 3 + c] = restFields[c * (K - 1) + (k - 1)].load(raw, rowStart: rowStart)
                }
            }
        }

        do {
            return try GaussianCloud(
                shDegree: shDegree,
                positions: positions,
                logScales: logScales,
                rotations: rotations,
                opacityLogits: opacityLogits,
                sh: sh
            )
        } catch {
            throw .unsupportedFormat("inconsistent data")
        }
    }
}

/// Scalar property types allowed in a PLY vertex element. All are converted to `Float`.
private enum ScalarType {
    case int8, uint8, int16, uint16, int32, uint32, float32, float64

    init?(name: String) {
        switch name {
        case "char", "int8": self = .int8
        case "uchar", "uint8": self = .uint8
        case "short", "int16": self = .int16
        case "ushort", "uint16": self = .uint16
        case "int", "int32": self = .int32
        case "uint", "uint32": self = .uint32
        case "float", "float32": self = .float32
        case "double", "float64": self = .float64
        default: return nil
        }
    }

    var size: Int {
        switch self {
        case .int8, .uint8: 1
        case .int16, .uint16: 2
        case .int32, .uint32, .float32: 4
        case .float64: 8
        }
    }

    @inline(__always)
    func read(_ raw: UnsafeRawBufferPointer, at offset: Int) -> Float {
        switch self {
        case .int8:
            Float(raw.loadUnaligned(fromByteOffset: offset, as: Int8.self))
        case .uint8:
            Float(raw.loadUnaligned(fromByteOffset: offset, as: UInt8.self))
        case .int16:
            Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: Int16.self)))
        case .uint16:
            Float(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
        case .int32:
            Float(Int32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: Int32.self)))
        case .uint32:
            Float(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
        case .float32:
            Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
        case .float64:
            Float(Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt64.self))))
        }
    }
}
