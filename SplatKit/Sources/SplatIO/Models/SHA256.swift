import Foundation

/// Incremental SHA-256 (FIPS 180-4), implemented in pure Swift so the model store
/// needs no third-party cryptography package.
public struct SHA256: Sendable {
    private static let roundConstants: [UInt32] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]

    private static let initialState: [UInt32] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]

    private static let blockSize = 64

    private var state: [UInt32] = SHA256.initialState
    /// Bytes of the current, not yet processed block. Always fewer than 64 between calls.
    private var pending: [UInt8] = []
    private var totalBytes: UInt64 = 0

    public init() {}

    /// Feeds more input into the hash. Can be called any number of times with chunks of any size.
    public mutating func update(_ data: Data) {
        totalBytes &+= UInt64(data.count)
        for byte in data {
            pending.append(byte)
            if pending.count == SHA256.blockSize {
                processBlock(pending)
                pending.removeAll(keepingCapacity: true)
            }
        }
    }

    /// Returns the 32-byte digest of everything fed so far. The hasher itself is not changed,
    /// so `update` may still be called afterwards.
    public func finalize() -> [UInt8] {
        var copy = self
        let bitLength: UInt64 = copy.totalBytes &* 8

        copy.pending.append(0x80)
        if copy.pending.count > 56 {
            while copy.pending.count < SHA256.blockSize {
                copy.pending.append(0)
            }
            copy.processBlock(copy.pending)
            copy.pending.removeAll(keepingCapacity: true)
        }
        while copy.pending.count < 56 {
            copy.pending.append(0)
        }
        var shift: Int = 56
        while shift >= 0 {
            copy.pending.append(UInt8(truncatingIfNeeded: bitLength >> UInt64(shift)))
            shift -= 8
        }
        copy.processBlock(copy.pending)
        copy.pending.removeAll(keepingCapacity: true)

        var digest: [UInt8] = []
        digest.reserveCapacity(32)
        for word in copy.state {
            digest.append(UInt8(truncatingIfNeeded: word >> 24))
            digest.append(UInt8(truncatingIfNeeded: word >> 16))
            digest.append(UInt8(truncatingIfNeeded: word >> 8))
            digest.append(UInt8(truncatingIfNeeded: word))
        }
        return digest
    }

    /// Lower-case hexadecimal digest of a complete buffer.
    public static func hexDigest(of data: Data) -> String {
        var hasher = SHA256()
        hasher.update(data)
        return hexString(hasher.finalize())
    }

    /// Lower-case hexadecimal rendering of a digest.
    public static func hexString(_ bytes: [UInt8]) -> String {
        let digits: [Character] = Array("0123456789abcdef")
        var result = ""
        result.reserveCapacity(bytes.count * 2)
        for byte in bytes {
            result.append(digits[Int(byte >> 4)])
            result.append(digits[Int(byte & 0x0f)])
        }
        return result
    }

    @inline(__always)
    private static func rotateRight(_ value: UInt32, _ count: UInt32) -> UInt32 {
        (value >> count) | (value << (32 - count))
    }

    /// Compresses one 64-byte block into `state`. `block` must hold exactly 64 bytes.
    private mutating func processBlock(_ block: [UInt8]) {
        var schedule = [UInt32](repeating: 0, count: 64)
        for index in 0..<16 {
            let base: Int = index * 4
            var word: UInt32 = 0
            word |= UInt32(block[base]) << 24
            word |= UInt32(block[base + 1]) << 16
            word |= UInt32(block[base + 2]) << 8
            word |= UInt32(block[base + 3])
            schedule[index] = word
        }
        for index in 16..<64 {
            let w15: UInt32 = schedule[index - 15]
            let w2: UInt32 = schedule[index - 2]
            let sigma0: UInt32 = SHA256.rotateRight(w15, 7) ^ SHA256.rotateRight(w15, 18) ^ (w15 >> 3)
            let sigma1: UInt32 = SHA256.rotateRight(w2, 17) ^ SHA256.rotateRight(w2, 19) ^ (w2 >> 10)
            schedule[index] = schedule[index - 16] &+ sigma0 &+ schedule[index - 7] &+ sigma1
        }

        var a: UInt32 = state[0]
        var b: UInt32 = state[1]
        var c: UInt32 = state[2]
        var d: UInt32 = state[3]
        var e: UInt32 = state[4]
        var f: UInt32 = state[5]
        var g: UInt32 = state[6]
        var h: UInt32 = state[7]

        for index in 0..<64 {
            let bigSigma1: UInt32 = SHA256.rotateRight(e, 6) ^ SHA256.rotateRight(e, 11) ^ SHA256.rotateRight(e, 25)
            let choose: UInt32 = (e & f) ^ (~e & g)
            let temp1: UInt32 = h &+ bigSigma1 &+ choose &+ SHA256.roundConstants[index] &+ schedule[index]
            let bigSigma0: UInt32 = SHA256.rotateRight(a, 2) ^ SHA256.rotateRight(a, 13) ^ SHA256.rotateRight(a, 22)
            let majority: UInt32 = (a & b) ^ (a & c) ^ (b & c)
            let temp2: UInt32 = bigSigma0 &+ majority

            h = g
            g = f
            f = e
            e = d &+ temp1
            d = c
            c = b
            b = a
            a = temp1 &+ temp2
        }

        state[0] = state[0] &+ a
        state[1] = state[1] &+ b
        state[2] = state[2] &+ c
        state[3] = state[3] &+ d
        state[4] = state[4] &+ e
        state[5] = state[5] &+ f
        state[6] = state[6] &+ g
        state[7] = state[7] &+ h
    }
}
