import Foundation

/// Why a depth map cannot be resampled to a view size.
public enum DepthResampleError: Error, Equatable {
    case aspectMismatch(source: SIMD2<Int>, target: SIMD2<Int>)
}

extension DepthMap {
    /// Relative aspect tolerance between the depth map and the target size (P3).
    static let aspectTolerance: Double = 0.01
    /// Largest max/min ratio of the four bilinear taps that is still interpolated (P3).
    static let interpolationRatio: Float = 1.1

    /// Resamples the map to `targetWidth` × `targetHeight` (P3 rules).
    ///
    /// The aspect ratios must agree within 1 % unless `allowAspectChange` is set (model output
    /// only). Values are bilinear only when all four taps are valid and their max/min ratio is at
    /// most 1.1, which keeps depth edges sharp. Otherwise the value is the valid tap nearest to the
    /// rounded position, falling back to the nearest valid tap (raster order breaks ties), else 0.
    /// Confidence and weights are taken from the nearest source pixel. Invalid pixels stay invalid.
    public func resampled(width targetWidth: Int,
                          height targetHeight: Int,
                          allowAspectChange: Bool = false) throws(DepthResampleError) -> DepthMap {
        if targetWidth == width && targetHeight == height {
            return self
        }
        if !allowAspectChange {
            let sourceAspect: Double = Double(width) / Double(height)
            let targetAspect: Double = Double(targetWidth) / Double(targetHeight)
            let relativeDifference: Double = abs(sourceAspect - targetAspect) / targetAspect
            guard relativeDifference <= DepthMap.aspectTolerance else {
                throw .aspectMismatch(source: SIMD2(width, height),
                                      target: SIMD2(targetWidth, targetHeight))
            }
        }

        let outputCount: Int = targetWidth * targetHeight
        var outputValues: [Float] = [Float](repeating: 0, count: outputCount)
        var outputConfidence: [UInt8]? = nil
        if confidence != nil {
            outputConfidence = [UInt8](repeating: 0, count: outputCount)
        }
        var outputWeights: [Float]? = nil
        if weights != nil {
            outputWeights = [Float](repeating: 0, count: outputCount)
        }

        let scaleX: Double = Double(width) / Double(targetWidth)
        let scaleY: Double = Double(height) / Double(targetHeight)
        for row in 0..<targetHeight {
            let sourceV: Double = (Double(row) + 0.5) * scaleY - 0.5
            for column in 0..<targetWidth {
                let sourceU: Double = (Double(column) + 0.5) * scaleX - 0.5
                let outputIndex: Int = row * targetWidth + column
                outputValues[outputIndex] = sampleValue(sourceU: sourceU, sourceV: sourceV)
                if let confidence, outputConfidence != nil {
                    let nearest: Int = nearestIndex(sourceU: sourceU, sourceV: sourceV)
                    outputConfidence![outputIndex] = confidence[nearest]
                }
                if let weights, outputWeights != nil {
                    let nearest: Int = nearestIndex(sourceU: sourceU, sourceV: sourceV)
                    outputWeights![outputIndex] = weights[nearest]
                }
            }
        }
        return DepthMap(width: targetWidth, height: targetHeight, kind: kind,
                        values: outputValues, confidence: outputConfidence, isEstimated: isEstimated,
                        weights: outputWeights)
    }

    /// Rotates the map 90° clockwise. The result is `height` wide and `width` high.
    public func rotatedClockwise() -> DepthMap {
        rotated(clockwise: true)
    }

    /// Rotates the map 90° counter-clockwise. The result is `height` wide and `width` high.
    public func rotatedCounterClockwise() -> DepthMap {
        rotated(clockwise: false)
    }

    // MARK: - Helpers

    /// Clamped source index of the pixel nearest to the given source coordinate.
    private func nearestIndex(sourceU: Double, sourceV: Double) -> Int {
        let roundedU: Double = sourceU.rounded()
        let roundedV: Double = sourceV.rounded()
        let column: Int = clampIndex(Int(roundedU), limit: width)
        let row: Int = clampIndex(Int(roundedV), limit: height)
        return row * width + column
    }

    /// Value at a source coordinate under the P3 rules. Returns 0 when no tap is valid.
    private func sampleValue(sourceU: Double, sourceV: Double) -> Float {
        let floorU: Double = sourceU.rounded(.down)
        let floorV: Double = sourceV.rounded(.down)
        let column0: Int = clampIndex(Int(floorU), limit: width)
        let column1: Int = clampIndex(Int(floorU) + 1, limit: width)
        let row0: Int = clampIndex(Int(floorV), limit: height)
        let row1: Int = clampIndex(Int(floorV) + 1, limit: height)

        // Raster order: top-left, top-right, bottom-left, bottom-right.
        let taps: [(column: Int, row: Int)] = [
            (column0, row0), (column1, row0), (column0, row1), (column1, row1),
        ]
        var tapIndices: [Int] = []
        var tapValid: [Bool] = []
        for tap in taps {
            let index: Int = tap.row * width + tap.column
            tapIndices.append(index)
            tapValid.append(isValid(index))
        }

        if tapValid.allSatisfy({ $0 }) {
            let a: Float = values[tapIndices[0]]
            let b: Float = values[tapIndices[1]]
            let c: Float = values[tapIndices[2]]
            let d: Float = values[tapIndices[3]]
            let largest: Float = max(max(a, b), max(c, d))
            let smallest: Float = min(min(a, b), min(c, d))
            if largest <= smallest * DepthMap.interpolationRatio {
                let weightX: Double = sourceU - floorU
                let weightY: Double = sourceV - floorV
                let top: Double = Double(a) * (1 - weightX) + Double(b) * weightX
                let bottom: Double = Double(c) * (1 - weightX) + Double(d) * weightX
                let blended: Double = top * (1 - weightY) + bottom * weightY
                return Float(blended)
            }
        }

        // Nearest valid tap by the rounded position first.
        let nearest: Int = nearestIndex(sourceU: sourceU, sourceV: sourceV)
        if isValid(nearest) {
            return values[nearest]
        }

        // Otherwise the valid tap closest to the exact position, raster order on ties.
        var bestValue: Float = 0
        var bestDistance: Double = Double.infinity
        for (position, tap) in taps.enumerated() where tapValid[position] {
            let deltaU: Double = Double(tap.column) - sourceU
            let deltaV: Double = Double(tap.row) - sourceV
            let distance: Double = deltaU * deltaU + deltaV * deltaV
            if distance < bestDistance {
                bestDistance = distance
                bestValue = values[tapIndices[position]]
            }
        }
        return bestValue
    }

    private func clampIndex(_ index: Int, limit: Int) -> Int {
        if index < 0 {
            return 0
        }
        if index >= limit {
            return limit - 1
        }
        return index
    }

    private func rotated(clockwise: Bool) -> DepthMap {
        let outputWidth: Int = height
        let outputHeight: Int = width
        let outputCount: Int = outputWidth * outputHeight
        var outputValues: [Float] = [Float](repeating: 0, count: outputCount)
        var outputConfidence: [UInt8]? = nil
        if confidence != nil {
            outputConfidence = [UInt8](repeating: 0, count: outputCount)
        }
        var outputWeights: [Float]? = nil
        if weights != nil {
            outputWeights = [Float](repeating: 0, count: outputCount)
        }

        for row in 0..<outputHeight {
            for column in 0..<outputWidth {
                // Clockwise: output (x, y) takes source (x = y', y = H - 1 - x).
                // Counter-clockwise: output (x, y) takes source (x = W - 1 - y, y = x).
                let sourceColumn: Int
                let sourceRow: Int
                if clockwise {
                    sourceColumn = row
                    sourceRow = height - 1 - column
                } else {
                    sourceColumn = width - 1 - row
                    sourceRow = column
                }
                let sourceIndex: Int = sourceRow * width + sourceColumn
                let outputIndex: Int = row * outputWidth + column
                outputValues[outputIndex] = values[sourceIndex]
                if let confidence, outputConfidence != nil {
                    outputConfidence![outputIndex] = confidence[sourceIndex]
                }
                if let weights, outputWeights != nil {
                    outputWeights![outputIndex] = weights[sourceIndex]
                }
            }
        }
        return DepthMap(width: outputWidth, height: outputHeight, kind: kind,
                        values: outputValues, confidence: outputConfidence, isEstimated: isEstimated,
                        weights: outputWeights)
    }
}
