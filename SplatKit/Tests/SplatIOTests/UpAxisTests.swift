import Testing
import SplatCore
@testable import SplatIO

@Suite struct UpAxisTests {

    @Test func upAxisFollowsTheCaptureSource() {
        #expect(UpAxis.forInputKind(.colmap) == .negativeY)
        #expect(UpAxis.forInputKind(nil) == .negativeY)
        #expect(UpAxis.forInputKind(.photos) == .positiveY)
        #expect(UpAxis.forInputKind(.video) == .positiveY)
        #expect(UpAxis.forInputKind(.arkit) == .positiveY)
    }
}
