import AudioKit
import AudioKitEX
import SoundpipeAudioKit
import SwiftUI
import AVFoundation

class MFCCExtractor: ObservableObject {
    private let engine = AudioEngine()
    private var mic: AudioEngine.InputNode?
    private var mixer: Mixer?
    private var fft: FFTTap?
    private var extractionTimer: Timer?
    private var latestFFTData: [Float]?
    
    // Use 20 MFCC coefficients
    private var mfccData: [Float] = Array(repeating: 0, count: 12)
    
    @Published var isRecording = false
    @Published var currentMFCCs: [Float] = Array(repeating: 0, count: 12)
    private let extractionInterval: Double = 2.0 // Fixed at 2 seconds
    
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
        
        // Setup FFT tap with default buffer size
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
                    
                    // Extract MFCCs every 2 seconds
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
        // 1. Apply mel filter bank
        let melFilteredData = applyMelFilterBank(fftData: fftData)
        
        // 2. Take log
        let logMelFilteredData = melFilteredData.map { log($0 + Float.leastNonzeroMagnitude) }
        
        // 3. Apply DCT (Discrete Cosine Transform)
        let mfccs = applyDCT(logMelData: logMelFilteredData)
        
        // 4. Keep the first 20 coefficients
        for i in 0..<min(12, mfccs.count) {
            mfccData[i] = mfccs[i]
        }
        
        // 5. Update the published property on the main thread
        DispatchQueue.main.async {
            self.currentMFCCs = self.mfccData
        }
    }
    
    private func applyMelFilterBank(fftData: [Float]) -> [Float] {
        // Simple mel filter bank implementation
        var melFilteredData = Array(repeating: Float(0), count: 26)
        
        // Simulate mel filtering (simplified)
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
        // Simplified DCT implementation
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

struct ContentView: View {
    @StateObject private var mfccExtractor = MFCCExtractor()
    @State private var timer: Timer?
    @State private var mfccValues: [Float] = Array(repeating: 0, count: 20)
    
    var body: some View {
        VStack(spacing: 20) {
            Text("MFCC Extractor")
                .font(.title)
                .padding()
            
            // Display current extraction status
            Text(mfccExtractor.isRecording ? "Recording..." : "Not Recording")
                .foregroundColor(mfccExtractor.isRecording ? .red : .gray)
                .font(.headline)
            
            // Start/Stop Button
            Button(action: {
                if mfccExtractor.isRecording {
                    stopRecording()
                } else {
                    startRecording()
                }
            }) {
                Text(mfccExtractor.isRecording ? "Stop" : "Start")
                    .padding()
                    .frame(width: 100)
                    .background(mfccExtractor.isRecording ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
            
            // Current MFCC values
            VStack(alignment: .leading) {
                Text("MFCC Values (Updated every 2 seconds):")
                    .font(.headline)
                ScrollView {
                    Text(mfccValuesString)
                        .font(.caption)
                        .monospaced()
                }
                .frame(height: 300)
            }
            .padding()
        }
        .padding()
        .onAppear {
            setupUIUpdateTimer()
        }
        .onDisappear {
            timer?.invalidate()
            mfccExtractor.stop()
        }
    }
    
    private var mfccValuesString: String {
        return mfccValues.enumerated().map { index, value in
            "MFCC[\(index)]: \(String(format: "%.6f", value))"
        }.joined(separator: "\n")
    }
    
    private func setupUIUpdateTimer() {
        // Update UI with MFCC values periodically
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            mfccValues = mfccExtractor.getCurrentMFCC()
        }
    }
    
    private func startRecording() {
        mfccExtractor.start()
    }
    
    private func stopRecording() {
        mfccExtractor.stop()
    }
}

// App entry point
@main
struct MFCCApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
