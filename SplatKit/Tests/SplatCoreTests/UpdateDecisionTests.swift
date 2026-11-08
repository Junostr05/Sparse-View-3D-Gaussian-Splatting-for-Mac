import Foundation
import Testing
import SplatCore

@Suite struct UpdateDecisionTests {
    private let releaseLink: String = "https://github.com/Junostr05/Sparse-View-3D-Gaussian-Splatting-for-Mac/releases/tag/v1.3.0"

    private func json(tag: String, link: String) -> Data {
        let text: String = "{\"tag_name\": \"\(tag)\", \"html_url\": \"\(link)\", \"prerelease\": false, \"draft\": false, \"name\": \"x\"}"
        return Data(text.utf8)
    }

    @Test func newerReleaseOffersUpdateWithoutLeadingV() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "v1.3.0", htmlURL: releaseLink, currentVersion: "1.2.0")
        #expect(outcome == .updateAvailable(latest: "1.3.0", url: URL(string: releaseLink)!))
    }

    @Test func equalOrOlderReleaseIsUpToDate() {
        let equal: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "v1.2.0", htmlURL: releaseLink, currentVersion: "1.2.0")
        #expect(equal == .upToDate(current: "1.2.0"))
        let older: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "1.1.0", htmlURL: releaseLink, currentVersion: "1.2.0")
        #expect(older == .upToDate(current: "1.2.0"))
    }

    @Test func preReleaseOfSameNumberIsNotAnUpdate() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "1.2.0-beta.1", htmlURL: releaseLink, currentVersion: "1.2.0")
        #expect(outcome == .upToDate(current: "1.2.0"))
    }

    @Test func unparsableTagFails() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "latest", htmlURL: releaseLink, currentVersion: "1.2.0")
        if case .failed = outcome {} else { Issue.record("expected failed, got \(outcome)") }
    }

    @Test func unparsableCurrentVersionFails() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "1.3.0", htmlURL: releaseLink, currentVersion: "?")
        if case .failed = outcome {} else { Issue.record("expected failed, got \(outcome)") }
    }

    @Test func releaseLinkMustBeHTTPSOnGitHub() {
        #expect(UpdateDecision.releaseURL(releaseLink) != nil)
        #expect(UpdateDecision.releaseURL("http://github.com/a/b") == nil)
        #expect(UpdateDecision.releaseURL("https://example.com/a/b") == nil)
        #expect(UpdateDecision.releaseURL("https://github.com.evil.example/a/b") == nil)
        #expect(UpdateDecision.releaseURL("https://user@github.com/a/b") == nil)
        #expect(UpdateDecision.releaseURL("javascript:alert(1)") == nil)
        #expect(UpdateDecision.releaseURL("") == nil)
    }

    @Test func badLinkFailsOnlyWhenAnUpdateIsOffered() {
        let newer: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "1.3.0", htmlURL: "https://example.com/x", currentVersion: "1.2.0")
        if case .failed = newer {} else { Issue.record("expected failed, got \(newer)") }
        let same: ReleaseCheckOutcome = UpdateDecision.evaluate(
            tagName: "1.2.0", htmlURL: "https://example.com/x", currentVersion: "1.2.0")
        #expect(same == .upToDate(current: "1.2.0"))
    }

    @Test func status404MeansNoReleases() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            statusCode: 404, body: Data("{\"message\": \"Not Found\"}".utf8), currentVersion: "1.2.0")
        #expect(outcome == .noReleases)
    }

    @Test func rateLimitStatusesFail() {
        for code in [403, 429] {
            let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
                statusCode: code, body: Data(), currentVersion: "1.2.0")
            if case .failed(let message) = outcome {
                #expect(message.contains("limiting"))
            } else {
                Issue.record("expected failed for \(code), got \(outcome)")
            }
        }
    }

    @Test func otherStatusFailsWithCode() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            statusCode: 500, body: Data(), currentVersion: "1.2.0")
        #expect(outcome == .failed("GitHub answered with status 500."))
    }

    @Test func malformedBodyFails() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            statusCode: 200, body: Data("<html>".utf8), currentVersion: "1.2.0")
        if case .failed = outcome {} else { Issue.record("expected failed, got \(outcome)") }
    }

    @Test func status200DecodesAndEvaluates() {
        let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
            statusCode: 200, body: json(tag: "v1.3.0", link: releaseLink), currentVersion: "1.2.0")
        #expect(outcome == .updateAvailable(latest: "1.3.0", url: URL(string: releaseLink)!))
    }

    @Test func decodesGitHubFieldNames() throws {
        let release: GitHubRelease = try JSONDecoder().decode(
            GitHubRelease.self, from: json(tag: "v1.0", link: releaseLink))
        #expect(release.tagName == "v1.0")
        #expect(release.htmlURL == releaseLink)
        #expect(release.prerelease == false)
    }
}
