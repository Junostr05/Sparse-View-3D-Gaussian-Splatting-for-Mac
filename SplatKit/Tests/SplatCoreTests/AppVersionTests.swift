import Testing
import SplatCore

@Suite struct AppVersionTests {
    private func version(_ text: String) -> AppVersion {
        guard let parsed = AppVersion(text) else {
            preconditionFailure("\(text) should parse")
        }
        return parsed
    }

    @Test func parsesFullAndShortForms() {
        let full: AppVersion = version("1.2.0")
        #expect(full.major == 1)
        #expect(full.minor == 2)
        #expect(full.patch == 0)
        #expect(full.prerelease.isEmpty)
        #expect(version("1") == version("1.0.0"))
        #expect(version("1.2") == version("1.2.0"))
    }

    @Test func leadingVIsOptional() {
        #expect(version("v1.2.0") == version("1.2.0"))
        #expect(version("V2") == version("2.0.0"))
    }

    @Test func surroundingWhitespaceIsIgnored() {
        #expect(version("  1.2.0 \n") == version("1.2.0"))
    }

    @Test func ordersNumericComponents() {
        let older: AppVersion = version("1.2.0")
        let newerPatch: AppVersion = version("1.2.1")
        let newerMinor: AppVersion = version("1.10.0")
        let newerMajor: AppVersion = version("2.0.0")
        #expect(older < newerPatch)
        #expect(newerPatch < newerMinor)
        #expect(newerMinor < newerMajor)
        #expect(!(newerMajor < older))
    }

    @Test func comparesNumbersNotText() {
        #expect(version("1.9.0") < version("1.10.0"))
    }

    @Test func equalVersionsAreEqual() {
        #expect(!(version("1.2.0") < version("1.2")))
        #expect(!(version("1.2") < version("1.2.0")))
        #expect(version("v1.2.0") == version("1.2.0"))
    }

    @Test func preReleaseSortsBelowSameNumericRelease() {
        #expect(version("1.2.0-beta.1") < version("1.2.0"))
        #expect(!(version("1.2.0") < version("1.2.0-beta.1")))
        #expect(version("1.1.9") < version("1.2.0-beta.1"))
    }

    @Test func preReleaseIdentifiersOrderLikeSemver() {
        #expect(version("1.2.0-alpha") < version("1.2.0-beta"))
        #expect(version("1.2.0-beta.2") < version("1.2.0-beta.10"))
        #expect(version("1.2.0-1") < version("1.2.0-alpha"))
        #expect(version("1.2.0-beta") < version("1.2.0-beta.1"))
    }

    @Test func rejectsGarbage() {
        let rejected: [String] = [
            "", "   ", "v", "abc", "1.2.3.4", "1..2", "1.2.", ".1", "1.x", "-1", "+1", "1.-2",
            "1.2.0-", "1.2.0-beta..1", "1.2.0-be ta", "1.2.0+build.5", "99999999999999999999999",
            "1.2.0-beta!", "1,2,0",
        ]
        for text in rejected {
            #expect(AppVersion(text) == nil, "\(text) should not parse")
        }
    }
}
