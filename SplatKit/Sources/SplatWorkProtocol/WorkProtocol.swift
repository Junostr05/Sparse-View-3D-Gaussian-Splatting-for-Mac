import Foundation
import SplatCore
import SplatIO

/// Framing of the work messages between the Mac and a companion device (an iPhone on the same network).
///
/// A frame is a 12-byte header followed by a payload:
/// - bytes 0..<4: magic `SWP1`
/// - bytes 4..<6: protocol version (UInt16, little-endian)
/// - byte 6: message kind (1 depth request, 2 depth response, 3 failure)
/// - byte 7: reserved, 0
/// - bytes 8..<12: payload length (UInt32, little-endian)
///
/// Payloads start with a UInt32 header length and a JSON header carrying the request id. A depth request continues with
/// the raw RGB8 pixels (width × height × 3 bytes). A depth response continues with a `.sdpt` depth map as written by
/// `DepthMapCodec`. A failure has only the JSON header, with a message.
public enum WorkProtocol {
    public static let version: UInt16 = 1
    public static let headerSize: Int = 12
    static let magic: [UInt8] = [0x53, 0x57, 0x50, 0x31]  // "SWP1"

    /// One depth request: an image for the named view, computed by the named model.
    public struct DepthRequest: Equatable, Sendable {
        public var id: String
        public var name: String
        public var model: String
        public var width: Int
        public var height: Int
        /// Row-major RGB8, `width × height × 3` bytes.
        public var rgb: [UInt8]

        public init(id: String, name: String, model: String, width: Int, height: Int, rgb: [UInt8]) {
            self.id = id
            self.name = name
            self.model = model
            self.width = width
            self.height = height
            self.rgb = rgb
        }
    }

    public enum Message: Equatable, Sendable {
        case depthRequest(DepthRequest)
        case depthResponse(id: String, map: DepthMap)
        case failure(id: String, message: String)
    }

    public enum Failure: Error, Equatable, Sendable, CustomStringConvertible {
        case truncated
        case badMagic
        case unsupportedVersion(UInt16)
        case unknownKind(UInt8)
        case badHeader
        case pixelCountMismatch(expected: Int, actual: Int)
        case badDepthMap(String)

        public var description: String {
            switch self {
            case .truncated: return "the message is shorter than it says"
            case .badMagic: return "not a SparseSplat work message"
            case .unsupportedVersion(let value): return "protocol version \(value) is not supported"
            case .unknownKind(let value): return "unknown message kind \(value)"
            case .badHeader: return "the message header is not valid"
            case .pixelCountMismatch(let expected, let actual):
                return "expected \(expected) pixel bytes, got \(actual)"
            case .badDepthMap(let reason): return "the depth map could not be read: \(reason)"
            }
        }
    }

    private struct RequestHeader: Codable {
        var id: String
        var name: String
        var model: String
        var width: Int
        var height: Int
    }

    private struct IdHeader: Codable {
        var id: String
        var message: String?
    }

    // MARK: - Encoding

    public static func encode(_ message: Message) -> Data {
        var kind: UInt8
        var payload: Data
        switch message {
        case .depthRequest(let request):
            kind = 1
            let header = RequestHeader(id: request.id, name: request.name, model: request.model,
                                       width: request.width, height: request.height)
            payload = jsonPrefixed(header)
            payload.append(contentsOf: request.rgb)
        case .depthResponse(let id, let map):
            kind = 2
            payload = jsonPrefixed(IdHeader(id: id, message: nil))
            payload.append(DepthMapCodec.encode(map))
        case .failure(let id, let text):
            kind = 3
            payload = jsonPrefixed(IdHeader(id: id, message: text))
        }
        var frame = Data(magic)
        appendUInt16(version, to: &frame)
        frame.append(kind)
        frame.append(0)
        appendUInt32(UInt32(payload.count), to: &frame)
        frame.append(payload)
        return frame
    }

    // MARK: - Decoding

    /// The payload length announced by the first `headerSize` bytes of a frame. Lets a reader know how many more
    /// bytes to wait for. Throws when the header is not a valid frame start.
    public static func payloadLength(header: Data) throws(Failure) -> Int {
        guard header.count >= headerSize else {
            throw .truncated
        }
        let bytes = [UInt8](header.prefix(headerSize))
        guard Array(bytes[0..<4]) == magic else {
            throw .badMagic
        }
        let value: UInt16 = UInt16(bytes[4]) | (UInt16(bytes[5]) << 8)
        guard value == version else {
            throw .unsupportedVersion(value)
        }
        return Int(readUInt32(bytes, at: 8))
    }

    /// Decodes one complete frame.
    public static func decode(_ frame: Data) throws(Failure) -> Message {
        let length: Int = try payloadLength(header: frame)
        guard frame.count == headerSize + length else {
            throw .truncated
        }
        let bytes = [UInt8](frame)
        let kind: UInt8 = bytes[6]
        let payload: [UInt8] = Array(bytes[headerSize...])
        guard payload.count >= 4 else {
            throw .badHeader
        }
        let headerLength: Int = Int(readUInt32(payload, at: 0))
        guard payload.count >= 4 + headerLength else {
            throw .badHeader
        }
        let headerData = Data(payload[4..<(4 + headerLength)])
        let rest: [UInt8] = Array(payload[(4 + headerLength)...])
        let decoder = JSONDecoder()

        switch kind {
        case 1:
            guard let header = try? decoder.decode(RequestHeader.self, from: headerData),
                  header.width > 0, header.height > 0 else {
                throw .badHeader
            }
            let expected: Int = header.width * header.height * 3
            guard rest.count == expected else {
                throw .pixelCountMismatch(expected: expected, actual: rest.count)
            }
            return .depthRequest(DepthRequest(id: header.id, name: header.name, model: header.model,
                                              width: header.width, height: header.height, rgb: rest))
        case 2:
            guard let header = try? decoder.decode(IdHeader.self, from: headerData) else {
                throw .badHeader
            }
            do {
                let map: DepthMap = try DepthMapCodec.decode(Data(rest))
                return .depthResponse(id: header.id, map: map)
            } catch {
                throw .badDepthMap(String(describing: error))
            }
        case 3:
            guard let header = try? decoder.decode(IdHeader.self, from: headerData) else {
                throw .badHeader
            }
            return .failure(id: header.id, message: header.message ?? "unknown error")
        default:
            throw .unknownKind(kind)
        }
    }

    // MARK: - Byte helpers

    private static func jsonPrefixed<T: Encodable>(_ value: T) -> Data {
        let json: Data = (try? JSONEncoder().encode(value)) ?? Data()
        var data = Data()
        appendUInt32(UInt32(json.count), to: &data)
        data.append(json)
        return data
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8(value >> 8))
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        for shift in stride(from: 0, to: 32, by: 8) {
            data.append(UInt8((value >> UInt32(shift)) & 0xFF))
        }
    }

    private static func readUInt32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(bytes[offset + index]) << UInt32(8 * index)
        }
        return value
    }
}
