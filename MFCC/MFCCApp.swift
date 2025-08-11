import AudioKit
import AudioKitEX
import SoundpipeAudioKit
import SwiftUI
import AVFoundation
import CoreML

class MFCCExtractor: ObservableObject {
    private let engine = AudioEngine()
    private var mic: AudioEngine.InputNode?
    private var mixer: Mixer?
    private var fft: FFTTap?
    private var extractionTimer: Timer?
    private var latestFFTData: [Float]?
    
    // Use 12 MFCC coefficients as you have now
    private var mfccData: [Float] = Array(repeating: 0, count: 12)
    
    @Published var isRecording = false
    @Published var currentMFCCs: [Float] = Array(repeating: 0, count: 12)
    
    // New: Publish prediction result
    @Published var predictionResult: String = "No prediction yet"
    
    private let extractionInterval: Double = 2.0 // 2 seconds
    
    // Core ML model instance
    let model = try! v1_0(configuration: MLModelConfiguration())
    
    init() {
        setupAudioSession()
        setupAudio()
    }
    
    private func setupAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord,
                                                          options: [.defaultToSpeaker, .allowBluetoothA2DP])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("Failed to set up audio session: \(error.localizedDescription)")
        }
    }
    
    private func setupAudio() {
        guard let input = engine.input else {
            print("Audio input not available")
            return
        }
        
        mic = input
        mixer = Mixer(input)
        
        guard let mixer = mixer else { return }
        engine.output = mixer
        
        fft = FFTTap(input) { fftData in
            self.latestFFTData = fftData
        }
    }
    
    func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async {
                completion(granted)
            }
        }
    }
    
    func start() {
        requestMicrophonePermission { granted in
            if granted {
                do {
                    try self.engine.start()
                    self.fft?.start()
                    self.isRecording = true
                    
                    self.extractionTimer = Timer.scheduledTimer(withTimeInterval: self.extractionInterval,
                                                              repeats: true) { _ in
                        if let latestFFTData = self.latestFFTData {
                            self.calculateMFCC(from: latestFFTData)
                        }
                    }
                } catch {
                    print("Failed to start audio engine: \(error.localizedDescription)")
                }
            } else {
                print("Microphone permission denied")
            }
        }
    }
    
    func stop() {
        engine.stop()
        fft?.stop()
        extractionTimer?.invalidate()
        extractionTimer = nil
        isRecording = false
    }
    
    func getCurrentMFCC() -> [Float] {
        return currentMFCCs
    }
    
    private func calculateMFCC(from fftData: [Float]) {
        let melFilteredData = applyMelFilterBank(fftData: fftData)
        let logMelFilteredData = melFilteredData.map { log($0 + Float.leastNonzeroMagnitude) }
        let mfccs = applyDCT(logMelData: logMelFilteredData)
        
        for i in 0..<min(12, mfccs.count) {
            mfccData[i] = mfccs[i]
        }
        
        DispatchQueue.main.async {
            self.currentMFCCs = self.mfccData
            self.runModelPrediction(with: self.mfccData)
        }
    }
    
    private func runModelPrediction(with mfccs: [Float]) {
        do {
            // Your model expects rank-2 MLMultiArray, so shape accordingly
            let mlArray = try MLMultiArray(shape: [1, NSNumber(value: mfccs.count)], dataType: .float32)
            
            for (i, val) in mfccs.enumerated() {
                mlArray[i] = NSNumber(value: val)
            }
            
            // Replace 'input' with your actual input parameter name
            let input = v1_0Input(x: mlArray)
            
            let output = try model.prediction(input: input)
            
            // var_10 is your output probabilities MLMultiArray
            let probs = output.var_10
            
            // Convert MLMultiArray to [Float]
            let floatArray = (0..<probs.count).map { Float(truncating: probs[$0]) }
            
            // Find max probability and its index
            if let maxIndex = floatArray.indices.max(by: { floatArray[$0] < floatArray[$1] }) {
                let maxProb = floatArray[maxIndex]
                DispatchQueue.main.async {
                    self.predictionResult = "Class index: \(maxIndex), Probability: \(String(format: "%.3f", maxProb))"
                }
            }
        } catch {
            DispatchQueue.main.async {
                self.predictionResult = "Prediction error: \(error.localizedDescription)"
            }
        }
    }
    
    // Your existing mel filter bank and DCT code here...
    private func applyMelFilterBank(fftData: [Float]) -> [Float] {
        var melFilteredData = Array(repeating: Float(0), count: 26)
        for i in 0..<melFilteredData.count {
            let startIdx = i * fftData.count / (melFilteredData.count * 2)
            let endIdx = min((i + 1) * fftData.count / (melFilteredData.count * 2), fftData.count - 1)
            var sum: Float = 0
            for j in startIdx..<endIdx {
                sum += fftData[j]
            }
            melFilteredData[i] = sum / Float(endIdx - startIdx)
        }
        return melFilteredData
    }
    
    private func applyDCT(logMelData: [Float]) -> [Float] {
        let n = logMelData.count
        var dctOutput = Array(repeating: Float(0), count: n)
        for k in 0..<n {
            var sum: Float = 0
            for i in 0..<n {
                let angle = Float.pi * Float(k) * (Float(i) + 0.5) / Float(n)
                sum += logMelData[i] * cos(angle)
            }
            dctOutput[k] = sum * 2.0 / Float(n)
        }
        return dctOutput
    }
}
