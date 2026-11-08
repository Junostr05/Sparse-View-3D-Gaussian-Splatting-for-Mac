import Foundation

/// A command the command palette can find. The app's command type conforms.
public protocol CommandSearchable {
    /// The name shown in the palette and matched first.
    var title: String { get }
    /// Extra words that match the command without appearing in its title.
    var keywords: [String] { get }
    /// Disabled commands are left out of the results.
    var isEnabled: Bool { get }
}

/// Fuzzy ranking for the command palette. Pure, so it runs and is tested on Linux.
///
/// Every whitespace-separated token of the query must match, in order and not necessarily adjacent,
/// somewhere in the title or in one of the keywords. Matching ignores case and diacritics. A token
/// scores higher when its characters start words, when they are adjacent and when it matches at the
/// start of the title. Matches in a keyword count half. Ties keep the registry order.
public enum CommandSearch {
    /// The enabled commands that match `query`, best match first.
    ///
    /// An empty query returns every enabled command in the order given.
    public static func rank<Command: CommandSearchable>(query: String, commands: [Command]) -> [Command] {
        let tokens: [String] = queryTokens(query)
        let enabled: [(index: Int, command: Command)] = commands.enumerated()
            .filter { $0.element.isEnabled }
            .map { (index: $0.offset, command: $0.element) }
        if tokens.isEmpty {
            return enabled.map { $0.command }
        }
        let wholeQuery: String = tokens.joined(separator: " ")
        var scored: [(score: Double, index: Int, command: Command)] = []
        for entry in enabled {
            guard let score = score(tokens: tokens, wholeQuery: wholeQuery, command: entry.command) else {
                continue
            }
            scored.append((score: score, index: entry.index, command: entry.command))
        }
        scored.sort { (lhs, rhs) -> Bool in
            if lhs.score != rhs.score {
                return lhs.score > rhs.score
            }
            return lhs.index < rhs.index
        }
        return scored.map { $0.command }
    }

    // MARK: - Scoring

    /// Score of a command for all tokens, or nil when some token matches nowhere.
    private static func score<Command: CommandSearchable>(tokens: [String], wholeQuery: String,
                                                          command: Command) -> Double? {
        let title: String = fold(command.title)
        let titleCharacters: [Character] = Array(title)
        let keywordCharacters: [[Character]] = command.keywords.map { Array(fold($0)) }
        var total: Double = 0
        for token in tokens {
            let tokenCharacters: [Character] = Array(token)
            var best: Double? = matchScore(tokenCharacters, in: titleCharacters)
            for keyword in keywordCharacters {
                if let keywordScore = matchScore(tokenCharacters, in: keyword) {
                    let halved: Double = 0.5 * keywordScore
                    if best == nil || halved > best! {
                        best = halved
                    }
                }
            }
            guard let tokenScore = best else {
                return nil
            }
            total += tokenScore
        }
        if title.hasPrefix(wholeQuery) {
            total += 10
        }
        return total
    }

    /// Score of `token` as a subsequence of `text`, matching each character at its leftmost free position.
    /// Nil when the token is not a subsequence.
    private static func matchScore(_ token: [Character], in text: [Character]) -> Double? {
        guard !token.isEmpty else { return nil }
        var positions: [Int] = []
        positions.reserveCapacity(token.count)
        var searchFrom: Int = 0
        for character in token {
            var position: Int = searchFrom
            while position < text.count, text[position] != character {
                position += 1
            }
            guard position < text.count else { return nil }
            positions.append(position)
            searchFrom = position + 1
        }
        var score: Double = 0
        for (k, position) in positions.enumerated() {
            // One point per matched character.
            score += 1
            // A character that starts a word.
            if position == 0 || !isWordCharacter(text[position - 1]) {
                score += 3
            }
            // A character directly after the previous match.
            if k > 0, position == positions[k - 1] + 1 {
                score += 2
            }
        }
        // The token starts the text.
        if positions[0] == 0 {
            score += 4
        }
        // Gaps between the first and the last matched character cost a little.
        let span: Int = positions[positions.count - 1] - positions[0] + 1 - token.count
        score -= 0.25 * Double(span)
        return score
    }

    private static func isWordCharacter(_ character: Character) -> Bool {
        character.isLetter || character.isNumber
    }

    // MARK: - Normalisation

    /// Lowercased and without diacritics, so that "Émigré" matches "emigre".
    private static func fold(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
    }

    private static func queryTokens(_ query: String) -> [String] {
        fold(query)
            .split(whereSeparator: { $0.isWhitespace })
            .map { String($0) }
    }
}
