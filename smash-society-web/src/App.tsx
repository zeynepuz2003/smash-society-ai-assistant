import { useState } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      // Arka planda bekleyen Python API'mizin kapısını çalıyoruz
      const response = await axios.post('http://127.0.0.1:8080/tahmin', { text });
      setResult(response.data.niyet);
    } catch (error) {
      console.error("API Hatası:", error);
      setResult("Bağlantı hatası: Python API arka planda çalışmıyor olabilir.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>SMASH SOCIETY</h1>
        <div className="subtitle">AI Customer Assistant</div>
      </div>

      <div className="content">
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label>Customer Message</label>
            <textarea
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Örn: My fries were completely cold when they arrived..."
            />
          </div>
          
          <button type="submit" className="submit-btn" disabled={loading || !text.trim()}>
            {loading ? 'Yapay Zeka Analiz Ediyor...' : 'Analyze Intent'}
          </button>
        </form>

        {result && (
          <div className="result-box">
            <div className="result-title">Target Department</div>
            <div className="result-intent">{result}</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;