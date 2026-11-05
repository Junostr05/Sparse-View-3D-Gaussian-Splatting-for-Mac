import Foundation
import Testing
@testable import SplatCore

/// Value of SH basis function `k` (channel 0) at `dir`, obtained by evaluating a degree-3
/// coefficient set that is zero except coefficient `k`, channel 0 = 1.
private func basis(_ k: Int, _ dir: SIMD3<Float>) -> Float {
    var coefficients = [Float](repeating: 0, count: 16 * 3)
    coefficients[k * 3] = 1
    return SphericalHarmonics.evaluate(degree: 3, coefficients: coefficients[...], direction: dir).x
}

@Test func basisIsOrthonormal() {
    let n = 20_000
    let weight = 4 * Double.pi / Double(n)
    let goldenAngle = Double.pi * (3 - 5.0.squareRoot())

    // Evaluate all 16 basis functions at each Fibonacci-sphere point once.
    var values = [[Double]](repeating: [Double](repeating: 0, count: 16), count: n)
    for i in 0..<n {
        let z = 1 - Double(2 * i + 1) / Double(n)
        let r = max(0, 1 - z * z).squareRoot()
        let phi = Double(i) * goldenAngle
        let dir = SIMD3<Float>(Float(r * cos(phi)), Float(r * sin(phi)), Float(z))
        for k in 0..<16 {
            values[i][k] = Double(basis(k, dir))
        }
    }

    var gram = [[Double]](repeating: [Double](repeating: 0, count: 16), count: 16)
    for i in 0..<16 {
        for j in 0..<16 {
            var sum = 0.0
            for p in 0..<n {
                sum += values[p][i] * values[p][j]
            }
            gram[i][j] = sum * weight
        }
    }

    var maxDiagonalError = 0.0
    var maxOffDiagonalError = 0.0
    for i in 0..<16 {
        for j in 0..<16 {
            if i == j {
                let err = abs(gram[i][j] - 1)
                maxDiagonalError = max(maxDiagonalError, err)
                #expect(err < 2e-3, "Gram diagonal (\(i), \(j)) = \(gram[i][j]), expected 1")
            } else {
                let err = abs(gram[i][j])
                maxOffDiagonalError = max(maxOffDiagonalError, err)
                #expect(err < 2e-3, "Gram off-diagonal (\(i), \(j)) = \(gram[i][j]), expected 0")
            }
        }
    }
    print("SH orthonormality: max diagonal error = \(maxDiagonalError), "
        + "max off-diagonal error = \(maxOffDiagonalError)")
}

@Test func degreeLimitTruncatesBands() {
    let raw = SIMD3<Float>(0.3, -0.5, 0.8)
    let dir = raw / (raw * raw).sum().squareRoot()
    let coefficients = [Float](repeating: 1, count: 16 * 3)

    let degree1 = SphericalHarmonics.evaluate(degree: 1, coefficients: coefficients[...], direction: dir).x
    let expected1 = (0..<4).reduce(Float(0)) { $0 + basis($1, dir) }
    #expect(abs(degree1 - expected1) < 1e-5,
            "degree 1: got \(degree1), expected \(expected1)")

    let degree2 = SphericalHarmonics.evaluate(degree: 2, coefficients: coefficients[...], direction: dir).x
    let expected2 = (0..<9).reduce(Float(0)) { $0 + basis($1, dir) }
    #expect(abs(degree2 - expected2) < 1e-5,
            "degree 2: got \(degree2), expected \(expected2)")
}
