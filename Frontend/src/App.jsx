import React, { useState } from 'react';
import axios from 'axios'; // Import axios
import './App.css'; // We'll create this for basic styling

function App() {
  const [question, setQuestion] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [humanizedResult, setHumanizedResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Backend API URL (FastAPI default)
  const API_URL = 'http://localhost:8000/humanize'; // Ensure this matches your FastAPI port

  const handleHumanize = async () => {
    if (!question.trim() || !aiResponse.trim()) {
      setError('Please enter both a question and an AI response.');
      return;
    }

    setIsLoading(true);
    setError('');
    setHumanizedResult(''); // Clear previous result

    try {
      console.log('Sending request to:', API_URL);
      const response = await axios.post(API_URL, {
        question: question,
        ai_response: aiResponse,
      });
      console.log('Received response:', response.data);
      setHumanizedResult(response.data.humanized_response);
    } catch (err) {
      console.error('API Error:', err);
      let errorMessage = 'Failed to humanize text. Please try again.';
      if (err.response) {
        // Server responded with a status code outside 2xx range
        errorMessage = `Error: ${err.response.status} - ${err.response.data?.detail || err.message}`;
      } else if (err.request) {
        // Request was made but no response received (backend down?)
        errorMessage = 'Error: Could not connect to the backend server. Is it running?';
      } else {
        // Something else happened
        errorMessage = `Error: ${err.message}`;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>AI Response Humanizer ✨</h1>
      <div className="input-section">
        <label htmlFor="question">Question:</label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter the original question..."
          rows={3}
          disabled={isLoading}
        />

        <label htmlFor="aiResponse">AI Response:</label>
        <textarea
          id="aiResponse"
          value={aiResponse}
          onChange={(e) => setAiResponse(e.target.value)}
          placeholder="Paste the AI's response here..."
          rows={8}
          disabled={isLoading}
        />

        <button onClick={handleHumanize} disabled={isLoading}>
          {isLoading ? 'Humanizing...' : 'Humanize!'}
        </button>
      </div>

      {error && <p className="error-message">{error}</p>}

      {humanizedResult && (
        <div className="result-section">
          <h2>Humanized Response:</h2>
          <p className="result-text">{humanizedResult}</p>
        </div>
      )}
    </div>
  );
}

export default App;
