import Foundation
import Vision
import AppKit

// Usage: swift ocr.swift <image> [minY] [maxY] [minX] [maxX]
// Crop region in normalized coordinates (0..1, origin bottom-left)
let args = CommandLine.arguments
let path = args[1]
let minY = args.count > 2 ? Double(args[2])! : 0.0
let maxY = args.count > 3 ? Double(args[3])! : 1.0
let minX = args.count > 4 ? Double(args[4])! : 0.0
let maxX = args.count > 5 ? Double(args[5])! : 1.0

guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    FileHandle.standardError.write("ERR cannot load image\n".data(using: .utf8)!)
    exit(1)
}

let req = VNRecognizeTextRequest { request, error in
    guard let obs = request.results as? [VNRecognizedTextObservation] else { return }
    let filtered = obs.filter { o in
        let b = o.boundingBox
        return b.midY >= minY && b.midY <= maxY && b.midX >= minX && b.midX <= maxX
    }
    let sorted = filtered.sorted { a, b in
        let ay = a.boundingBox.midY, by = b.boundingBox.midY
        if abs(ay - by) > 0.015 { return ay > by }
        return a.boundingBox.minX < b.boundingBox.minX
    }
    var currentY: CGFloat = -9
    for o in sorted {
        let b = o.boundingBox
        if abs(b.midY - currentY) > 0.015 {
            print("__ROW__ y=\(String(format: "%.3f", b.midY))")
        }
        if let t = o.topCandidates(1).first { print("\(String(format: "%.3f", b.minX))  \(t.string)") }
        currentY = b.midY
    }
}
req.recognitionLevel = .accurate
req.usesLanguageCorrection = false

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try handler.perform([req])
