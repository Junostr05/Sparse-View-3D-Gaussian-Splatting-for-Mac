/// A release version such as "1.2.0", "v1.2" or "1.2.0-beta.1".
///
/// One to three numeric components are accepted, a leading "v" is optional, and missing components count as 0.
/// A "-suffix" marks a pre-release, which sorts below the same numeric version. Build metadata ("+...") and
/// anything else that does not parse make `init?` return nil.
public struct AppVersion: Comparable, Equatable, Sendable {
    public let major: Int
    public let minor: Int
    public let patch: Int
    /// Dot-separated pre-release identifiers, empty for a release.
    public let prerelease: [String]

    public init?(_ text: String) {
        var body: Substring = Substring(text.trimmingWhitespace())
        if body.first == "v" || body.first == "V" {
            body = body.dropFirst()
        }

        let core: Substring
        let suffix: Substring?
        if let dash = body.firstIndex(of: "-") {
            core = body[body.startIndex..<dash]
            suffix = body[body.index(after: dash)...]
        } else {
            core = body
            suffix = nil
        }

        let components: [Substring] = core.split(separator: ".", omittingEmptySubsequences: false)
        guard (1...3).contains(components.count) else { return nil }
        var numbers: [Int] = []
        for component in components {
            guard let number = AppVersion.parseNumber(component) else { return nil }
            numbers.append(number)
        }
        while numbers.count < 3 {
            numbers.append(0)
        }

        var identifiers: [String] = []
        if let suffix {
            let parts: [Substring] = suffix.split(separator: ".", omittingEmptySubsequences: false)
            guard !parts.isEmpty else { return nil }
            for part in parts {
                guard !part.isEmpty, part.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }) else {
                    return nil
                }
                identifiers.append(String(part))
            }
        }

        self.major = numbers[0]
        self.minor = numbers[1]
        self.patch = numbers[2]
        self.prerelease = identifiers
    }

    public static func < (lhs: AppVersion, rhs: AppVersion) -> Bool {
        if lhs.major != rhs.major { return lhs.major < rhs.major }
        if lhs.minor != rhs.minor { return lhs.minor < rhs.minor }
        if lhs.patch != rhs.patch { return lhs.patch < rhs.patch }
        // A release sorts above any pre-release of the same numbers.
        if lhs.prerelease.isEmpty || rhs.prerelease.isEmpty {
            return !lhs.prerelease.isEmpty && rhs.prerelease.isEmpty
        }
        let count: Int = min(lhs.prerelease.count, rhs.prerelease.count)
        for index in 0..<count {
            let left: String = lhs.prerelease[index]
            let right: String = rhs.prerelease[index]
            if left == right { continue }
            let leftNumber: Int? = AppVersion.parseNumber(Substring(left))
            let rightNumber: Int? = AppVersion.parseNumber(Substring(right))
            switch (leftNumber, rightNumber) {
            case let (a?, b?):
                return a < b
            case (_?, nil):
                // Numeric identifiers sort below alphanumeric ones.
                return true
            case (nil, _?):
                return false
            case (nil, nil):
                return left < right
            }
        }
        return lhs.prerelease.count < rhs.prerelease.count
    }

    /// Digits only, no sign. Returns nil for empty or non-digit text.
    private static func parseNumber(_ text: Substring) -> Int? {
        guard !text.isEmpty, text.allSatisfy({ $0.isASCII && $0.isNumber }) else { return nil }
        return Int(text)
    }
}

private extension String {
    func trimmingWhitespace() -> String {
        var scalars: Substring = Substring(self)
        while let first = scalars.first, first.isWhitespace { scalars = scalars.dropFirst() }
        while let last = scalars.last, last.isWhitespace { scalars = scalars.dropLast() }
        return String(scalars)
    }
}
