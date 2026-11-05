func approxEqual(_ a: Float, _ b: Float, tol: Float) -> Bool {
    abs(a - b) <= tol
}

func approxEqual(_ a: SIMD2<Float>, _ b: SIMD2<Float>, tol: Float) -> Bool {
    let d = a - b
    return abs(d.x) <= tol && abs(d.y) <= tol
}

func approxEqual(_ a: SIMD3<Float>, _ b: SIMD3<Float>, tol: Float) -> Bool {
    let d = a - b
    return abs(d.x) <= tol && abs(d.y) <= tol && abs(d.z) <= tol
}
