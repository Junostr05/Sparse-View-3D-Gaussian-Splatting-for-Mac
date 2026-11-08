import Foundation
import Observation
import SplatCore

/// Asks GitHub whether a newer release exists. It runs only when the user presses "Check for Updates", never on its own.
/// The request carries no personal data: a generic User-Agent, no cookies and no cache.
@Observable
final class UpdateChecker {
    enum State: Equatable {
        case idle
        case checking
        case upToDate(current: String)
        case updateAvailable(latest: String, url: URL)
        case noReleases
        case failed(String)
    }

    typealias Fetch = @Sendable (URLRequest) async throws -> (Data, URLResponse)

    private(set) var state: State = .idle

    @ObservationIgnored private let fetch: Fetch

    static let latestReleaseURL: URL = URL(
        string: "https://api.github.com/repos/Junostr05/Sparse-View-3D-Gaussian-Splatting-for-Mac/releases/latest"
    )!

    /// `fetch` defaults to an ephemeral URLSession. Tests can pass canned responses instead.
    init(fetch: @escaping Fetch = UpdateChecker.liveFetch) {
        self.fetch = fetch
    }

    /// Checks the latest published release against `currentVersion` (the bundle's short version string).
    func check(currentVersion: String) async {
        guard state != .checking else { return }
        state = .checking

        var request: URLRequest = URLRequest(url: Self.latestReleaseURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("SparseSplat", forHTTPHeaderField: "User-Agent")

        do {
            let (data, response) = try await fetch(request)
            guard let http = response as? HTTPURLResponse else {
                state = .failed("GitHub returned an unexpected response.")
                return
            }
            let outcome: ReleaseCheckOutcome = UpdateDecision.evaluate(
                statusCode: http.statusCode, body: data, currentVersion: currentVersion)
            state = Self.state(for: outcome)
        } catch {
            if error is CancellationError || (error as? URLError)?.code == .cancelled {
                state = .idle
                return
            }
            state = .failed(Self.message(for: error))
        }
    }

    private static func state(for outcome: ReleaseCheckOutcome) -> State {
        switch outcome {
        case .upToDate(let current):
            return .upToDate(current: current)
        case .updateAvailable(let latest, let url):
            return .updateAvailable(latest: latest, url: url)
        case .noReleases:
            return .noReleases
        case .failed(let message):
            return .failed(message)
        }
    }

    private static func message(for error: Error) -> String {
        guard let urlError = error as? URLError else {
            return "The update check failed."
        }
        switch urlError.code {
        case .notConnectedToInternet, .networkConnectionLost, .dataNotAllowed,
             .internationalRoamingOff, .cannotConnectToHost, .cannotFindHost:
            return "No internet connection. Check your network and try again."
        case .timedOut:
            return "GitHub did not answer in time. Try again later."
        default:
            return "The update check failed."
        }
    }

    /// No cookies, no cache and a 10 second limit, so the request leaves no trace on the device.
    private nonisolated static let ephemeralSession: URLSession = {
        let configuration: URLSessionConfiguration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 10
        configuration.timeoutIntervalForResource = 10
        configuration.httpShouldSetCookies = false
        configuration.httpCookieAcceptPolicy = .never
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }()

    nonisolated static func liveFetch(_ request: URLRequest) async throws -> (Data, URLResponse) {
        try await ephemeralSession.data(for: request)
    }
}
