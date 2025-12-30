from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Important for web
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import find_peaks
import warnings
import os
import io
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'voice-analyzer-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'aac', 'flac'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class VoiceAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.analysis_data = {}
    
    def analyze_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            if audio is None or len(audio) == 0:
                return False
            
            audio = audio / np.max(np.abs(audio) + 1e-10)
            
            target_samples = 3 * self.sample_rate
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                padded_audio = np.zeros(target_samples)
                padded_audio[:len(audio)] = audio
                audio = padded_audio
            
            self.analysis_data = {
                'audio': audio,
                'sample_rate': sr,
                'duration': len(audio) / sr,
                'file_path': file_path,
                'filename': os.path.basename(file_path)
            }
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def extract_spectral_features(self):
        audio = self.analysis_data['audio']
        sr = self.analysis_data['sample_rate']
        
        Sxx = np.abs(librosa.stft(audio))
        Sxx_db = librosa.amplitude_to_db(Sxx, ref=np.max)
        
        times = librosa.times_like(Sxx_db, sr=sr)
        freqs = librosa.fft_frequencies(sr=sr)
        
        prominent_points = []
        for i in range(Sxx_db.shape[1]):
            peaks, _ = find_peaks(Sxx_db[:, i], height=-40, distance=10)
            for peak in peaks:
                if freqs[peak] <= 5000:
                    prominent_points.append({
                        'time': float(times[i]),
                        'frequency': float(freqs[peak]),
                        'intensity': float(Sxx_db[peak, i])
                    })
        
        prominent_points.sort(key=lambda x: x['intensity'], reverse=True)
        self.analysis_data['prominent_points'] = prominent_points[:50]
        self.analysis_data['spectrogram'] = (freqs, times, Sxx_db)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        self.analysis_data['spectral_features'] = {
            'centroid': spectral_centroid.tolist(),
            'rolloff': spectral_rolloff.tolist()
        }
    
    def generate_plots(self):
        audio = self.analysis_data['audio']
        sr = self.analysis_data['sample_rate']
        prominent_points = self.analysis_data['prominent_points']
        freqs, times, Sxx_db = self.analysis_data['spectrogram']
        
        plots = {}
        
        try:
            # Waveform
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(audio, sr=sr, alpha=0.8, color='blue')
            plt.title('Audio Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plots['waveform'] = self.plot_to_base64()
            plt.close()
            
            # Spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(Sxx_db, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
            plt.title('Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plots['spectrogram'] = self.plot_to_base64()
            plt.close()
            
            # Frequency distribution
            plt.figure(figsize=(10, 4))
            if prominent_points:
                freqs_hist = [point['frequency'] for point in prominent_points]
                plt.hist(freqs_hist, bins=30, alpha=0.7, color='green', edgecolor='black')
                plt.title('Frequency Distribution')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plots['histogram'] = self.plot_to_base64()
            plt.close()
            
        except Exception as e:
            print(f"Plot error: {e}")
        
        return plots
    
    def plot_to_base64(self):
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def classify_voice(self):
        prominent_points = self.analysis_data['prominent_points']
        
        if not prominent_points or len(prominent_points) < 10:
            return "ANALYSIS INCONCLUSIVE", "Not enough features"
        
        frequencies = [point['frequency'] for point in prominent_points]
        
        human_ranges = [
            (80, 300, 1.0),
            (300, 1000, 0.8),
            (1000, 3000, 0.6),
            (2000, 4000, 0.4)
        ]
        
        weighted_human_score = 0
        for freq in frequencies:
            for low, high, weight in human_ranges:
                if low <= freq <= high:
                    weighted_human_score += weight
                    break
        
        human_ratio = weighted_human_score / len(frequencies)
        human_ratio = max(0, min(1, human_ratio))
        confidence = int(human_ratio * 100)
        mean_freq = np.mean(frequencies)
        
        if human_ratio > 0.75:
            result = "HUMAN VOICE"
            details = f"High confidence: {confidence}%"
        elif human_ratio > 0.6:
            result = "LIKELY HUMAN VOICE"
            details = f"Medium confidence: {confidence}%"
        elif human_ratio > 0.4:
            result = "UNCERTAIN"
            details = f"Mixed characteristics"
        elif human_ratio > 0.2:
            result = "POSSIBLY AI VOICE"
            details = f"Limited human features"
        else:
            result = "PROBABLY AI VOICE"
            details = f"Low human characteristics"
        
        self.analysis_data['classification'] = {
            'result': result,
            'confidence': confidence,
            'mean_frequency': float(mean_freq)
        }
        
        return result, details
    
    def get_statistics(self):
        points = self.analysis_data.get('prominent_points', [])
        
        if not points:
            return {}
        
        frequencies = [p['frequency'] for p in points]
        
        return {
            'total_points': len(points),
            'mean_frequency': float(np.mean(frequencies)),
            'min_frequency': float(np.min(frequencies)),
            'max_frequency': float(np.max(frequencies)),
            'duration': float(self.analysis_data.get('duration', 0))
        }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Analyzer</title>
        <style>
            body { font-family: Arial; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .upload-btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .upload-btn:hover { background: #45a049; }
            input[type="file"] { padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice Analyzer</h1>
            <p>Upload audio to detect if it's Human or AI-generated</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".wav,.mp3,.m4a,.aac,.flac" required>
                <br><br>
                <button type="submit" class="upload-btn">Analyze Voice</button>
            </form>
            <p><small>Supports: WAV, MP3, M4A, AAC, FLAC (max 16MB)</small></p>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file selected', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = VoiceAnalyzer()
        
        if not analyzer.analyze_audio(filepath):
            os.remove(filepath)
            return 'Error analyzing audio', 400
        
        analyzer.extract_spectral_features()
        result, details = analyzer.classify_voice()
        plots = analyzer.generate_plots()
        stats = analyzer.get_statistics()
        classification = analyzer.analysis_data.get('classification', {})
        
        # Simple results page
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Results</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                .result-box {{ padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .human {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
                .ai {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
                .uncertain {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
                .stats {{ background: #e9ecef; padding: 15px; border-radius: 5px; }}
                .back-btn {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Results</h1>
                <h3>File: {analyzer.analysis_data['filename']}</h3>
                
                <div class="result-box {'human' if 'HUMAN' in result else 'ai' if 'AI' in result else 'uncertain'}">
                    <h2>Result: {result}</h2>
                    <p>{details}</p>
                    <p>Confidence: {classification.get('confidence', 0)}%</p>
                </div>
                
                <div class="stats">
                    <h3>Statistics</h3>
                    <p>Duration: {stats.get('duration', 0):.2f} seconds</p>
                    <p>Prominent Points: {stats.get('total_points', 0)}</p>
                    <p>Mean Frequency: {stats.get('mean_frequency', 0):.1f} Hz</p>
                    <p>Frequency Range: {stats.get('min_frequency', 0):.1f} - {stats.get('max_frequency', 0):.1f} Hz</p>
                </div>
                
                <h3>Visualizations</h3>
        '''
        
        if 'waveform' in plots:
            html += f'''
                <h4>Waveform</h4>
                <img src="data:image/png;base64,{plots['waveform']}" alt="Waveform">
            '''
        
        if 'spectrogram' in plots:
            html += f'''
                <h4>Spectrogram</h4>
                <img src="data:image/png;base64,{plots['spectrogram']}" alt="Spectrogram">
            '''
        
        if 'histogram' in plots:
            html += f'''
                <h4>Frequency Distribution</h4>
                <img src="data:image/png;base64,{plots['histogram']}" alt="Frequency Distribution">
            '''
        
        html += '''
                <br>
                <a href="/" class="back-btn">Analyze Another File</a>
            </div>
        </body>
        </html>
        '''
        
        return html
    
    return 'Invalid file type', 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)