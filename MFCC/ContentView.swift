//
//  ContentView.swift
//  MFCC
//
//  Created by iOS Lab on 24/04/25.
import SwiftUI

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
            
            // New: Show prediction result
            VStack(alignment: .leading) {
                Text("Model Prediction:")
                    .font(.headline)
                Text(mfccExtractor.predictionResult)
                    .font(.body)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(white: 0.95))
                    .cornerRadius(8)
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
