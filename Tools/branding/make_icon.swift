// Renders the SparseSplat app icon and logo with CoreGraphics.
//
//   swift Tools/branding/make_icon.swift <output directory>
//
// Writes icon_1024.png (macOS icon artwork with its rounded body and shadow), mark_1024.png (the mark alone
// on a transparent background) and logo.png (mark and wordmark). `Tools/branding/build_iconset.sh` turns
// icon_1024.png into the asset catalog.
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

let colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

func makeContext(width: Int, height: Int) -> CGContext {
    let context: CGContext = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                       space: colorSpace,
                                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setAllowsAntialiasing(true)
    context.interpolationQuality = .high
    return context
}

func writePNG(_ context: CGContext, to url: URL) {
    let image: CGImage = context.makeImage()!
    let destination: CGImageDestination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
}

func color(_ hex: UInt32, alpha: CGFloat = 1) -> CGColor {
    let red: CGFloat = CGFloat((hex >> 16) & 0xFF) / 255
    let green: CGFloat = CGFloat((hex >> 8) & 0xFF) / 255
    let blue: CGFloat = CGFloat(hex & 0xFF) / 255
    return CGColor(colorSpace: colorSpace, components: [red, green, blue, alpha])!
}

/// One soft Gaussian: a radial gradient on a rotated ellipse, added to what is already there.
func splat(_ context: CGContext, x: CGFloat, y: CGFloat, rx: CGFloat, ry: CGFloat, degrees: CGFloat,
           rgb: UInt32, alpha: CGFloat, core: CGFloat = 0.0) {
    context.saveGState()
    context.setBlendMode(.plusLighter)
    context.translateBy(x: x, y: y)
    context.rotate(by: degrees * .pi / 180)
    context.scaleBy(x: rx, y: ry)
    let colors: [CGColor] = [color(rgb, alpha: alpha), color(rgb, alpha: alpha * 0.55), color(rgb, alpha: 0)]
    let locations: [CGFloat] = [core, 0.45, 1]
    let gradient: CGGradient = CGGradient(colorsSpace: colorSpace, colors: colors as CFArray, locations: locations)!
    context.drawRadialGradient(gradient, startCenter: .zero, startRadius: 0, endCenter: .zero, endRadius: 1,
                               options: [])
    context.restoreGState()
}

/// Deterministic pseudo random numbers, so that the artwork is reproducible.
struct Generator {
    var state: UInt64 = 0x9E3779B97F4A7C15
    mutating func next() -> CGFloat {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return CGFloat((state >> 33) & 0xFFFFFF) / CGFloat(0xFFFFFF)
    }
}

/// The mark, centred at (0, 0) in a 1000 x 1000 design space. `withBrackets` adds the viewfinder corners.
func drawMark(_ context: CGContext, withBrackets: Bool) {
    // Big glowing Gaussians.
    splat(context, x: -150, y: -70, rx: 330, ry: 120, degrees: 28, rgb: 0xFF2E93, alpha: 0.95)
    splat(context, x: 140, y: 40, rx: 340, ry: 115, degrees: -32, rgb: 0x19D3FF, alpha: 0.95)
    splat(context, x: 10, y: 170, rx: 250, ry: 105, degrees: 84, rgb: 0xFFB020, alpha: 0.9)
    splat(context, x: -40, y: -190, rx: 220, ry: 90, degrees: -70, rgb: 0x7B5CFF, alpha: 0.9)
    // Bright centre where they meet.
    splat(context, x: 0, y: 0, rx: 140, ry: 140, degrees: 0, rgb: 0xFFFFFF, alpha: 0.7, core: 0.1)

    // Many small splats around: a sparse point cloud.
    var generator = Generator()
    let palette: [UInt32] = [0xFF2E93, 0x19D3FF, 0xFFB020, 0x7B5CFF, 0xFFFFFF]
    for _ in 0..<70 {
        let angle: CGFloat = generator.next() * 2 * .pi
        let radius: CGFloat = 150 + generator.next() * 330
        let size: CGFloat = 9 + generator.next() * 24
        let tilt: CGFloat = generator.next() * 180
        let rgb: UInt32 = palette[Int(generator.next() * CGFloat(palette.count)) % palette.count]
        splat(context, x: cos(angle) * radius * 0.95, y: sin(angle) * radius * 0.72, rx: size * 1.5, ry: size,
              degrees: tilt, rgb: rgb, alpha: 0.65 + generator.next() * 0.3, core: 0.2)
    }

    if withBrackets {
        context.saveGState()
        context.setBlendMode(.normal)
        context.setStrokeColor(color(0xFFFFFF, alpha: 0.92))
        context.setLineWidth(26)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        let half: CGFloat = 395
        let arm: CGFloat = 120
        for sx in [CGFloat(-1), 1] {
            for sy in [CGFloat(-1), 1] {
                context.move(to: CGPoint(x: sx * half, y: sy * (half - arm)))
                context.addLine(to: CGPoint(x: sx * half, y: sy * half))
                context.addLine(to: CGPoint(x: sx * (half - arm), y: sy * half))
                context.strokePath()
            }
        }
        context.restoreGState()
    }
}

func squircle(_ rect: CGRect) -> CGPath {
    // macOS icon corner radius is about 22.4 % of the body width.
    CGPath(roundedRect: rect, cornerWidth: rect.width * 0.2237, cornerHeight: rect.width * 0.2237, transform: nil)
}

func renderIcon(size: Int) -> CGContext {
    let context: CGContext = makeContext(width: size, height: size)
    let scale: CGFloat = CGFloat(size) / 1024
    context.scaleBy(x: scale, y: scale)
    let body: CGRect = CGRect(x: 100, y: 100, width: 824, height: 824)

    // Drop shadow under the body.
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -14), blur: 28, color: color(0x000000, alpha: 0.45))
    context.setFillColor(color(0x0A0B1E))
    context.addPath(squircle(body))
    context.fillPath()
    context.restoreGState()

    // Body: deep indigo to near black.
    context.saveGState()
    context.addPath(squircle(body))
    context.clip()
    let background: CGGradient = CGGradient(colorsSpace: colorSpace,
                                            colors: [color(0x1F2459), color(0x0B0C26), color(0x050510)] as CFArray,
                                            locations: [0, 0.55, 1])!
    context.drawLinearGradient(background, start: CGPoint(x: body.minX, y: body.maxY),
                               end: CGPoint(x: body.maxX, y: body.minY), options: [])

    // The mark, a little smaller than the body.
    context.saveGState()
    context.translateBy(x: 512, y: 512)
    context.scaleBy(x: 0.74, y: 0.74)
    drawMark(context, withBrackets: true)
    context.restoreGState()

    // Soft light from the top, like glass.
    let sheen: CGGradient = CGGradient(colorsSpace: colorSpace,
                                       colors: [color(0xFFFFFF, alpha: 0.16), color(0xFFFFFF, alpha: 0)] as CFArray,
                                       locations: [0, 1])!
    context.drawLinearGradient(sheen, start: CGPoint(x: 512, y: body.maxY), end: CGPoint(x: 512, y: 560), options: [])
    context.restoreGState()

    // Thin edge highlight.
    context.saveGState()
    context.addPath(squircle(body.insetBy(dx: 1.5, dy: 1.5)))
    context.setStrokeColor(color(0xFFFFFF, alpha: 0.14))
    context.setLineWidth(3)
    context.strokePath()
    context.restoreGState()
    return context
}

func renderMark(size: Int) -> CGContext {
    let context: CGContext = makeContext(width: size, height: size)
    let scale: CGFloat = CGFloat(size) / 1000
    context.scaleBy(x: scale, y: scale)
    context.translateBy(x: 500, y: 500)
    context.scaleBy(x: 0.9, y: 0.9)
    drawMark(context, withBrackets: true)
    return context
}

func renderLogo() -> CGContext {
    let width: Int = 2400
    let height: Int = 900
    let context: CGContext = makeContext(width: width, height: height)
    // Dark rounded panel so that the glow reads on any page.
    context.setFillColor(color(0x0A0B1E))
    context.addPath(CGPath(roundedRect: CGRect(x: 0, y: 0, width: width, height: height),
                           cornerWidth: 120, cornerHeight: 120, transform: nil))
    context.fillPath()
    context.saveGState()
    context.translateBy(x: 450, y: 450)
    context.scaleBy(x: 0.78, y: 0.78)
    drawMark(context, withBrackets: true)
    context.restoreGState()

    func text(_ string: String, size: CGFloat, weight: CGFloat, x: CGFloat, y: CGFloat, alpha: CGFloat) {
        let font: CTFont = CTFontCreateWithName("SF Pro Display" as CFString, size, nil)
        let bold: CTFont = CTFontCreateCopyWithSymbolicTraits(font, size, nil, weight > 0.5 ? .traitBold : [], .traitBold) ?? font
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): bold,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color(0xFFFFFF, alpha: alpha),
        ]
        let line: CTLine = CTLineCreateWithAttributedString(NSAttributedString(string: string, attributes: attributes))
        context.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(line, context)
    }
    text("SparseSplat", size: 250, weight: 1, x: 900, y: 470, alpha: 1)
    text("Gaussian splats from a few photos", size: 78, weight: 0, x: 910, y: 290, alpha: 0.7)
    return context
}

let arguments: [String] = CommandLine.arguments
guard arguments.count >= 2 else {
    print("usage: make_icon.swift <output directory>")
    exit(2)
}
let output: URL = URL(fileURLWithPath: arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
writePNG(renderIcon(size: 1024), to: output.appending(path: "icon_1024.png"))
writePNG(renderMark(size: 1024), to: output.appending(path: "mark_1024.png"))
writePNG(renderLogo(), to: output.appending(path: "logo.png"))
print("wrote icon_1024.png, mark_1024.png and logo.png to \(output.path)")
