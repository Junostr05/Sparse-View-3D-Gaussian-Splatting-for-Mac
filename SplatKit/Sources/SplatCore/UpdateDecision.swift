import Foundation

/// The fields of GitHub's "latest release" response that the update check uses.
public struct GitHubRelease: Decodable, Equatable, Sendable {
    public let tagName: String
    public let htmlURL: String
    /// Not used for the decision: `/releases/latest` never returns drafts or pre-releases.
    public let prerelease: Bool?

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case htmlURL = "html_url"
        case prerelease
    }
}

/// What the update check tells the user.
public enum ReleaseCheckOutcome: Equatable, Sendable {
    case upToDate(current: String)
    case updateAvailable(latest: String, url: URL)
    /// GitHub answered 404 for `/releases/latest`: the repository has no published release yet.
    case noReleases
    /// A short, readable message for the user.
    case failed(String)
}

/// Pure decision logic for the "Check for Updates" button. The app performs the request and passes the result here.
public enum UpdateDecision {
    /// Interprets one HTTP answer from `/releases/latest` and compares it with the running version.
    public static func evaluate(statusCode: Int, body: Data, currentVersion: String) -> ReleaseCheckOutcome {
        switch statusCode {
        case 200:
            break
        case 404:
            return .noReleases
        case 403, 429:
            return .failed("GitHub is limiting requests right now. Try again later.")
        default:
            return .failed("GitHub answered with status \(statusCode).")
        }
        let release: GitHubRelease
        do {
            release = try JSONDecoder().decode(GitHubRelease.self, from: body)
        } catch {
            return .failed("The release information could not be read.")
        }
        return evaluate(tagName: release.tagName, htmlURL: release.htmlURL, currentVersion: currentVersion)
    }

    /// Compares the release tag with the running version. The release link is only checked when an update is offered.
    public static func evaluate(tagName: String, htmlURL: String, currentVersion: String) -> ReleaseCheckOutcome {
        guard let current = AppVersion(currentVersion) else {
            return .failed("This build's version (\(currentVersion)) cannot be compared.")
        }
        guard let latest = AppVersion(tagName) else {
            return .failed("The latest release has an unrecognized version tag.")
        }
        guard latest > current else {
            return .upToDate(current: currentVersion)
        }
        guard let url = releaseURL(htmlURL) else {
            return .failed("The release link is not a GitHub address.")
        }
        return .updateAvailable(latest: displayVersion(tagName), url: url)
    }

    /// The release link, only if it is an https URL on github.com without user information.
    public static func releaseURL(_ text: String) -> URL? {
        guard let url = URL(string: text),
              url.scheme?.lowercased() == "https",
              url.host?.lowercased() == "github.com",
              url.user == nil,
              url.password == nil else {
            return nil
        }
        return url
    }

    /// The tag without a leading "v", for display.
    static func displayVersion(_ tagName: String) -> String {
        let trimmed: String = tagName.trimmingPrefixVersionMarker()
        return trimmed
    }
}

private extension String {
    func trimmingPrefixVersionMarker() -> String {
        guard let first = first, first == "v" || first == "V" else { return self }
        return String(dropFirst())
    }
}
