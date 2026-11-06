import Foundation
import Testing
@testable import SplatIO

@Test func sha256OfEmptyInput() {
    #expect(SHA256.hexDigest(of: Data()) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
}

@Test func sha256OfABC() {
    #expect(SHA256.hexDigest(of: Data("abc".utf8)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
}

@Test func sha256OfFortyEightByteMessage() {
    let message = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    #expect(SHA256.hexDigest(of: Data(message.utf8)) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
}

@Test func sha256OfQuickBrownFox() {
    let message = "The quick brown fox jumps over the lazy dog"
    #expect(SHA256.hexDigest(of: Data(message.utf8)) == "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592")
}

@Test func sha256OfOneMillionAs() {
    let data = Data(repeating: 0x61, count: 1_000_000)
    #expect(SHA256.hexDigest(of: data) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
}

@Test func sha256ChunkedUpdatesMatchOneShot() {
    var payload = Data()
    for index in 0..<300 {
        payload.append(UInt8(truncatingIfNeeded: index * 7 + 3))
    }
    let oneShot = SHA256.hexDigest(of: payload)

    let chunkSizes: [Int] = [1, 63, 64, 65]
    for chunkSize in chunkSizes {
        var hasher = SHA256()
        var offset: Int = 0
        while offset < payload.count {
            let end: Int = min(offset + chunkSize, payload.count)
            hasher.update(payload.subdata(in: offset..<end))
            offset = end
        }
        #expect(SHA256.hexString(hasher.finalize()) == oneShot, "chunk size \(chunkSize)")
    }
}

@Test func sha256FinalizeDoesNotConsumeHasher() {
    var hasher = SHA256()
    hasher.update(Data("ab".utf8))
    _ = hasher.finalize()
    hasher.update(Data("c".utf8))
    #expect(SHA256.hexString(hasher.finalize()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
}
