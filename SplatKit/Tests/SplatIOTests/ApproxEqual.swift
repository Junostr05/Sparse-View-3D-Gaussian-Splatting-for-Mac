/// Element-wise comparison of two float arrays; false when the lengths differ.
func approxEqual(_ a: [Float], _ b: [Float], tol: Float) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= tol }
}
