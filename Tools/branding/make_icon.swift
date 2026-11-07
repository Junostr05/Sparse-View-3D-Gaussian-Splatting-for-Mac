// Generates the SparseSplat app icon (a Liquid Glass `.icon` package made of two dot layers) and the logo files.
//
//   swift Tools/branding/make_icon.swift <app icon output directory> <docs assets directory>
//
// Writes <app icon output directory>/AppIcon.icon (layers `front` and `back`, rendered with Liquid Glass by the
// system when the app is built), and <docs assets directory>/mark.png and logo.png (flat renderings).
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

let colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

func makeContext(width: Int, height: Int) -> CGContext {
    let context: CGContext = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                       space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.interpolationQuality = .high
    return context
}

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(colorSpace: colorSpace,
            components: [CGFloat((hex >> 16) & 255) / 255, CGFloat((hex >> 8) & 255) / 255,
                         CGFloat(hex & 255) / 255, alpha])!
}

func writePNG(_ context: CGContext, to path: String) {
    let destination: CGImageDestination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                                         UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, context.makeImage()!, nil)
    CGImageDestinationFinalize(destination)
}

struct Dot {
    var x: CGFloat
    var y: CGFloat
    var radius: CGFloat
    var hex: UInt32
    var alpha: CGFloat
}

func mix(_ a: UInt32, _ b: UInt32, _ t: CGFloat) -> UInt32 {
    func channel(_ value: UInt32, _ shift: UInt32) -> CGFloat { CGFloat((value >> shift) & 255) }
    let u: CGFloat = max(0, min(1, t))
    let red: UInt32 = UInt32(channel(a, 16) + (channel(b, 16) - channel(a, 16)) * u)
    let green: UInt32 = UInt32(channel(a, 8) + (channel(b, 8) - channel(a, 8)) * u)
    let blue: UInt32 = UInt32(channel(a, 0) + (channel(b, 0) - channel(a, 0)) * u)
    return (red << 16) | (green << 8) | blue
}

func drawDots(_ context: CGContext, _ dots: [Dot]) {
    for dot in dots {
        context.setFillColor(color(dot.hex, dot.alpha))
        context.fillEllipse(in: CGRect(x: dot.x - dot.radius, y: dot.y - dot.radius,
                                       width: 2 * dot.radius, height: 2 * dot.radius))
    }
}

/// A sphere made of latitude and longitude dots, seen from slightly above and to the side. Front dots are
/// large and bright, back dots small and faint; the colour runs from cyan at the top to magenta at the bottom.
func sphereDots(front: Bool) -> [Dot] {
    var dots: [Dot] = []
    let sphereRadius: CGFloat = 360
    let yaw: CGFloat = 0.5
    let pitch: CGFloat = 0.42
    for latitudeDegrees in stride(from: 15, through: 165, by: 15) {
        let latitude: CGFloat = CGFloat(latitudeDegrees) * .pi / 180
        let count: Int = max(8, Int(36 * sin(latitude)))
        for index in 0..<count {
            let longitude: CGFloat = CGFloat(index) / CGFloat(count) * 2 * .pi
            let px: CGFloat = sphereRadius * sin(latitude) * cos(longitude)
            let py: CGFloat = sphereRadius * cos(latitude)
            let pz: CGFloat = sphereRadius * sin(latitude) * sin(longitude)
            let x1: CGFloat = px * cos(yaw) + pz * sin(yaw)
            let z1: CGFloat = -px * sin(yaw) + pz * cos(yaw)
            let y2: CGFloat = py * cos(pitch) - z1 * sin(pitch)
            let z2: CGFloat = py * sin(pitch) + z1 * cos(pitch)
            if (z2 >= 0) != front { continue }
            let depth: CGFloat = (z2 + sphereRadius) / (2 * sphereRadius)
            let height: CGFloat = (y2 + sphereRadius) / (2 * sphereRadius)
            dots.append(Dot(x: x1, y: y2, radius: 10 + 18 * depth, hex: mix(0xFF3DDB, 0x18D4FF, height),
                            alpha: front ? 1 : 0.7))
        }
    }
    return dots
}

let arguments: [String] = CommandLine.arguments
guard arguments.count >= 3 else {
    print("usage: make_icon.swift <app icon output directory> <docs assets directory>")
    exit(2)
}
let iconDirectory: String = arguments[1] + "/AppIcon.icon"
let assetsDirectory: String = arguments[2]
try? FileManager.default.removeItem(atPath: iconDirectory)
try FileManager.default.createDirectory(atPath: iconDirectory + "/Assets", withIntermediateDirectories: true)
try FileManager.default.createDirectory(atPath: assetsDirectory, withIntermediateDirectories: true)

// --- App icon: two transparent 1024 x 1024 layers, origin in the centre.
func layerContext() -> CGContext {
    let context: CGContext = makeContext(width: 1024, height: 1024)
    context.translateBy(x: 512, y: 512)
    return context
}
let frontContext: CGContext = layerContext()
drawDots(frontContext, sphereDots(front: true))
writePNG(frontContext, to: iconDirectory + "/Assets/front.png")
let backContext: CGContext = layerContext()
drawDots(backContext, sphereDots(front: false))
writePNG(backContext, to: iconDirectory + "/Assets/back.png")

func layerJSON(_ name: String) -> [String: Any] {
    ["glass": true, "image-name": name + ".png", "name": name,
     "position": ["scale": 1, "translation-in-points": [0, 0]]]
}
func groupJSON(_ name: String, translucency: Double) -> [String: Any] {
    ["layers": [layerJSON(name)], "name": name, "lighting": "individual", "specular": true,
     "shadow": ["kind": "neutral", "opacity": 0.5],
     "translucency": ["enabled": true, "value": translucency]]
}
// The first group is the top one.
let iconJSON: [String: Any] = [
    "fill": ["automatic-gradient": "extended-srgb:0.03000,0.05000,0.22000,1.00000"],
    "groups": [groupJSON("front", translucency: 0.35), groupJSON("back", translucency: 0.3)],
    "supported-platforms": ["squares": ["macOS"]],
]
try JSONSerialization.data(withJSONObject: iconJSON, options: [.prettyPrinted, .sortedKeys])
    .write(to: URL(fileURLWithPath: iconDirectory + "/icon.json"))

// --- Flat renderings for documents: the mark alone, and the mark with the name.
func drawMark(_ context: CGContext) {
    drawDots(context, sphereDots(front: false))
    drawDots(context, sphereDots(front: true))
}
let mark: CGContext = makeContext(width: 1024, height: 1024)
mark.translateBy(x: 512, y: 512)
mark.scaleBy(x: 1.05, y: 1.05)
drawMark(mark)
writePNG(mark, to: assetsDirectory + "/mark.png")

let logoWidth: Int = 2400
let logoHeight: Int = 900
let logo: CGContext = makeContext(width: logoWidth, height: logoHeight)
logo.setFillColor(color(0x080D38))
logo.addPath(CGPath(roundedRect: CGRect(x: 0, y: 0, width: logoWidth, height: logoHeight),
                    cornerWidth: 120, cornerHeight: 120, transform: nil))
logo.fillPath()
logo.saveGState()
logo.translateBy(x: 470, y: 450)
logo.scaleBy(x: 0.9, y: 0.9)
drawMark(logo)
logo.restoreGState()
func text(_ string: String, size: CGFloat, bold: Bool, x: CGFloat, y: CGFloat, alpha: CGFloat) {
    let base: CTFont = CTFontCreateWithName("Helvetica Neue" as CFString, size, nil)
    let font: CTFont = bold ? (CTFontCreateCopyWithSymbolicTraits(base, size, nil, .traitBold, .traitBold) ?? base) : base
    let attributes: [NSAttributedString.Key: Any] = [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color(0xFFFFFF, alpha),
    ]
    let line: CTLine = CTLineCreateWithAttributedString(NSAttributedString(string: string, attributes: attributes))
    logo.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(line, logo)
}
text("SparseSplat", size: 250, bold: true, x: 930, y: 470, alpha: 1)
text("Gaussian splats from a few photos", size: 78, bold: false, x: 940, y: 290, alpha: 0.7)
writePNG(logo, to: assetsDirectory + "/logo.png")
print("wrote \(iconDirectory), \(assetsDirectory)/mark.png and \(assetsDirectory)/logo.png")
